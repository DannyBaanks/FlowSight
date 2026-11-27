const std = @import("std");
const assert = std.debug.assert;

const GRID_W: usize = 12;
const GRID_H: usize = 16;
const GRID_LEN: usize = GRID_W * GRID_H; // 192
const SIMD_I8_CHUNKS: usize = GRID_LEN / 32; // 6 chunks of 32 bytes (AVX2)
const Vec32i8 = @Vector(32, i8);
const Vec32i16 = @Vector(32, i16);
const Vec32i32 = @Vector(32, i32);
const QUANT_SCALE: f32 = 256.0;
const INV_QUANT_SQ: f32 = 1.0 / (QUANT_SCALE * QUANT_SCALE);

comptime {
    assert(GRID_LEN == 192);
    assert(GRID_LEN % 32 == 0);
}

const TemplateMeta = struct {
    codepoint: u32,
    aspect: f32,
    y_top: f32,
    y_bot: f32,
    holes: u8,
};

const TemplateVec = [SIMD_I8_CHUNKS]Vec32i8;

const Atlas = struct {
    meta: []TemplateMeta,
    coarse: []Vec32i8,
    vecs: []TemplateVec,
    has_coarse: bool,

    fn deinit(self: Atlas, gpa: std.mem.Allocator) void {
        gpa.free(self.meta);
        if (self.coarse.len > 0) gpa.free(self.coarse);
        gpa.free(self.vecs);
    }
};

const Comp = struct {
    x0: u32,
    y0: u32,
    x1: u32, // exclusive
    y1: u32, // exclusive
    area: u32,
    merged: bool = false,

    fn width(self: Comp) u32 {
        return self.x1 - self.x0;
    }

    fn height(self: Comp) u32 {
        return self.y1 - self.y0;
    }

    fn cy(self: Comp) f32 {
        return (@as(f32, @floatFromInt(self.y0)) + @as(f32, @floatFromInt(self.y1))) * 0.5;
    }
};

const PgmImage = struct {
    width: u32,
    height: u32,
    pixels: []const u8,
};

const OcrLine = struct {
    x0: u32,
    y0: u32,
    x1: u32,
    y1: u32,
    mean_conf: f32,
    min_conf: f32,
    text: []u8,
};

fn readU16Le(bytes: []const u8, offset: usize) u16 {
    return std.mem.readInt(u16, bytes[offset..][0..2], .little);
}

fn readI16Le(bytes: []const u8, offset: usize) i16 {
    return std.mem.readInt(i16, bytes[offset..][0..2], .little);
}

fn readU32Le(bytes: []const u8, offset: usize) u32 {
    return std.mem.readInt(u32, bytes[offset..][0..4], .little);
}

fn readF32Le(bytes: []const u8, offset: usize) f32 {
    const bits = readU32Le(bytes, offset);
    return @bitCast(bits);
}

fn quantizeF32ToI8(v: f32) i8 {
    const scaled = std.math.clamp(v * QUANT_SCALE, -127.0, 127.0);
    return @intFromFloat(@round(scaled));
}

const SortEntry = struct {
    meta: TemplateMeta,
    vec: TemplateVec,
};

fn sortEntryLessByAspect(_: void, a: SortEntry, b: SortEntry) bool {
    return a.meta.aspect < b.meta.aspect;
}

fn loadAtlas(gpa: std.mem.Allocator, raw: []const u8) !Atlas {
    if (raw.len < 16) return error.InvalidAtlas;
    if (!std.mem.eql(u8, raw[0..4], "FSOA")) return error.InvalidAtlasMagic;
    const version = readU16Le(raw, 4);
    const gw = readU16Le(raw, 6);
    const gh = readU16Le(raw, 8);
    const count = readU32Le(raw, 12);
    if ((version != 1 and version != 2 and version != 3) or gw != GRID_W or gh != GRID_H) return error.UnsupportedAtlas;

    // Fastest path: FSOA v3 is pre-sorted by aspect, SoA quantized i8 + 4x8 coarse filter.
    if (version == 3) {
        const meta_bytes = @as(usize, count) * 12;
        const coarse_bytes = @as(usize, count) * 32;
        const vecs_bytes = @as(usize, count) * GRID_LEN;
        const expected_len = 16 + meta_bytes + coarse_bytes + vecs_bytes;
        if (raw.len < expected_len) return error.TruncatedAtlas;

        const meta = try gpa.alloc(TemplateMeta, count);
        errdefer gpa.free(meta);
        const coarse = try gpa.alloc(Vec32i8, count);
        errdefer gpa.free(coarse);
        const vecs = try gpa.alloc(TemplateVec, count);
        errdefer gpa.free(vecs);

        var pos: usize = 16;
        for (meta) |*m| {
            const cp = readU32Le(raw, pos);
            const aspect_q8 = readU16Le(raw, pos + 4);
            const y_top_q8 = readI16Le(raw, pos + 6);
            const y_bot_q8 = readI16Le(raw, pos + 8);
            const holes = raw[pos + 10];
            pos += 12;
            m.* = .{
                .codepoint = cp,
                .aspect = @as(f32, @floatFromInt(aspect_q8)) / 256.0,
                .y_top = @as(f32, @floatFromInt(y_top_q8)) / 256.0,
                .y_bot = @as(f32, @floatFromInt(y_bot_q8)) / 256.0,
                .holes = holes,
            };
        }
        @memcpy(std.mem.sliceAsBytes(coarse), raw[pos .. pos + coarse_bytes]);
        pos += coarse_bytes;
        @memcpy(std.mem.sliceAsBytes(vecs), raw[pos .. pos + vecs_bytes]);
        return .{ .meta = meta, .coarse = coarse, .vecs = vecs, .has_coarse = true };
    }

    // Fast path: FSOA v2 is pre-sorted by aspect and pre-quantized to SoA i8.
    if (version == 2) {
        const meta_bytes = @as(usize, count) * 12;
        const vecs_bytes = @as(usize, count) * GRID_LEN;
        const expected_len = 16 + meta_bytes + vecs_bytes;
        if (raw.len < expected_len) return error.TruncatedAtlas;

        const meta = try gpa.alloc(TemplateMeta, count);
        errdefer gpa.free(meta);
        const vecs = try gpa.alloc(TemplateVec, count);
        errdefer gpa.free(vecs);

        var pos: usize = 16;
        for (meta) |*m| {
            const cp = readU32Le(raw, pos);
            const aspect_q8 = readU16Le(raw, pos + 4);
            const y_top_q8 = readI16Le(raw, pos + 6);
            const y_bot_q8 = readI16Le(raw, pos + 8);
            const holes = raw[pos + 10];
            pos += 12;
            m.* = .{
                .codepoint = cp,
                .aspect = @as(f32, @floatFromInt(aspect_q8)) / 256.0,
                .y_top = @as(f32, @floatFromInt(y_top_q8)) / 256.0,
                .y_bot = @as(f32, @floatFromInt(y_bot_q8)) / 256.0,
                .holes = holes,
            };
        }
        @memcpy(std.mem.sliceAsBytes(vecs), raw[pos .. pos + vecs_bytes]);
        return .{ .meta = meta, .coarse = &.{}, .vecs = vecs, .has_coarse = false };
    }

    const entry_bytes: usize = 12 + GRID_LEN * 4;
    const expected_len: usize = 16 + @as(usize, count) * entry_bytes;
    if (raw.len < expected_len) return error.TruncatedAtlas;

    const tmp = try gpa.alloc(SortEntry, count);
    defer gpa.free(tmp);

    var pos: usize = 16;
    for (tmp) |*e| {
        const cp = readU32Le(raw, pos);
        const aspect_q8 = readU16Le(raw, pos + 4);
        const y_top_q8 = readI16Le(raw, pos + 6);
        const y_bot_q8 = readI16Le(raw, pos + 8);
        const holes = raw[pos + 10];
        pos += 12;

        var vecs: TemplateVec = undefined;
        for (0..SIMD_I8_CHUNKS) |chunk| {
            var arr: [32]i8 = undefined;
            for (0..32) |k| {
                arr[k] = quantizeF32ToI8(readF32Le(raw, pos));
                pos += 4;
            }
            vecs[chunk] = arr;
        }
        e.* = .{
            .meta = .{
                .codepoint = cp,
                .aspect = @as(f32, @floatFromInt(aspect_q8)) / 256.0,
                .y_top = @as(f32, @floatFromInt(y_top_q8)) / 256.0,
                .y_bot = @as(f32, @floatFromInt(y_bot_q8)) / 256.0,
                .holes = holes,
            },
            .vec = vecs,
        };
    }
    std.mem.sort(SortEntry, tmp, {}, sortEntryLessByAspect);

    const meta = try gpa.alloc(TemplateMeta, count);
    errdefer gpa.free(meta);
    const vecs = try gpa.alloc(TemplateVec, count);
    errdefer gpa.free(vecs);
    for (tmp, 0..) |e, i| {
        meta[i] = e.meta;
        vecs[i] = e.vec;
    }
    return .{ .meta = meta, .coarse = &.{}, .vecs = vecs, .has_coarse = false };
}

fn skipPgmWsAndComments(data: []const u8, idx: *usize) void {
    while (idx.* < data.len) {
        const c = data[idx.*];
        if (c == '#') {
            while (idx.* < data.len and data[idx.*] != '\n') : (idx.* += 1) {}
        } else if (c == ' ' or c == '\t' or c == '\r' or c == '\n') {
            idx.* += 1;
        } else {
            break;
        }
    }
}

fn parsePgmUint(data: []const u8, idx: *usize) !u32 {
    skipPgmWsAndComments(data, idx);
    const start = idx.*;
    while (idx.* < data.len and data[idx.*] >= '0' and data[idx.*] <= '9') : (idx.* += 1) {}
    if (idx.* == start) return error.InvalidPgmHeader;
    return std.fmt.parseInt(u32, data[start..idx.*], 10);
}

fn parsePgm(data: []const u8) !PgmImage {
    if (data.len < 8 or data[0] != 'P' or data[1] != '5') return error.InvalidPgmMagic;
    var idx: usize = 2;
    const w = try parsePgmUint(data, &idx);
    const h = try parsePgmUint(data, &idx);
    const maxv = try parsePgmUint(data, &idx);
    if (maxv != 255 or w == 0 or h == 0) return error.UnsupportedPgm;
    if (idx >= data.len) return error.TruncatedPgm;
    // Skip single whitespace byte after maxval
    idx += 1;
    const needed = @as(usize, w) * @as(usize, h);
    if (data.len - idx < needed) return error.TruncatedPgm;
    return .{
        .width = w,
        .height = h,
        .pixels = data[idx .. idx + needed],
    };
}

const FgWorkerCtx = struct {
    img: PgmImage,
    fg: []u8,
    mask: []u8,
    tw: u32,
    th: u32,
    next_ty: std.atomic.Value(u32),
};

/// Minimum largest-1px-step / tile-luma-range ratio (percent) for a tile to keep foreground.
const SHARP_STEP_PCT: u32 = 30;

fn fgWorkerRun(ctx: *FgWorkerCtx) void {
    const tile: u32 = 16;
    const img = ctx.img;
    const fg = ctx.fg;
    const mask = ctx.mask;
    const width = img.width;
    const height = img.height;
    const tw = ctx.tw;
    const th = ctx.th;
    const Vec16u8 = @Vector(16, u8);
    const zeros16: Vec16u8 = @splat(0);
    const ones16: Vec16u8 = @splat(1);

    while (true) {
        const ty_start = ctx.next_ty.fetchAdd(4, .monotonic);
        if (ty_start >= th) break;
        const ty_end = @min(ty_start + 4, th);

        var ty = ty_start;
        while (ty < ty_end) : (ty += 1) {
            const y_start = ty * tile;
            const y_end = @min(y_start + tile, height);
            const rows = y_end - y_start;

            var tx: u32 = 0;
            while (tx < tw) : (tx += 1) {
                const x_start = tx * tile;
                const x_end = @min(x_start + tile, width);
                const cols = x_end - x_start;

                // 16-bin histogram to find dominant background luma in tile
                var hist_cnt: [16]u32 = .{0} ** 16;
                var hist_sum: [16]u32 = .{0} ** 16;
                // Largest 1-px luma step (horizontal or vertical) inside the tile: glyph strokes are
                // 1-3 px wide, so their edges jump a large fraction of the stroke contrast in one pixel.
                var max_step: u32 = 0;
                var tile_lo: u32 = 255;
                var tile_hi: u32 = 0;

                var y = y_start;
                while (y < y_end) : (y += 1) {
                    const row_off = @as(usize, y) * @as(usize, width) + x_start;
                    const row = img.pixels[row_off .. row_off + cols];
                    const prev_row: ?[]const u8 = if (y > y_start) img.pixels[row_off - width .. row_off - width + cols] else null;
                    for (row, 0..) |p, xi| {
                        const bin: usize = p >> 4;
                        hist_cnt[bin] += 1;
                        hist_sum[bin] += p;
                        if (p < tile_lo) tile_lo = p;
                        if (p > tile_hi) tile_hi = p;
                        if (xi > 0) {
                            const s: u32 = @abs(@as(i32, p) - @as(i32, row[xi - 1]));
                            if (s > max_step) max_step = s;
                        }
                        if (prev_row) |pr| {
                            const s: u32 = @abs(@as(i32, p) - @as(i32, pr[xi]));
                            if (s > max_step) max_step = s;
                        }
                    }
                }

                var best_bin: usize = 0;
                var best_cnt: u32 = 0;
                for (0..16) |b| {
                    if (hist_cnt[b] > best_cnt) {
                        best_cnt = hist_cnt[b];
                        best_bin = b;
                    }
                }
                const bg_u8: u8 = if (best_cnt > 0) @intCast(hist_sum[best_bin] / best_cnt) else 0;

                // Background coherence gate: real text over solid/gradient UI backgrounds concentrates
                // >=35% of tile pixels in the single dominant histogram bin (width 16 luma);
                // uniform RGB random noise (Irwin-Hall luma) peaks at ~15-22% in a single bin.
                const tile_px: u32 = rows * cols;
                // Edge-sharpness gate: smooth textures (blurred photos, wallpapers, upscaled noise) span a
                // large luma range but only through gradual ramps, so no single 1-px step reaches a
                // meaningful fraction of that range. Text strokes always do.
                const tile_range: u32 = tile_hi - tile_lo;
                const smooth_texture = tile_range >= 36 and max_step * 100 < tile_range * SHARP_STEP_PCT;
                if (best_cnt * 100 < tile_px * 35 or smooth_texture) {
                    var ry_z: u32 = 0;
                    while (ry_z < rows) : (ry_z += 1) {
                        const row_off = @as(usize, y_start + ry_z) * @as(usize, width) + x_start;
                        @memset(fg[row_off .. row_off + cols], 0);
                        @memset(mask[row_off .. row_off + cols], 0);
                    }
                    continue;
                }

                if (cols == 16) {
                    const bg_v: Vec16u8 = @splat(bg_u8);
                    var tile_fg: [16]Vec16u8 = undefined;
                    var max_v: Vec16u8 = zeros16;

                    var ry: u32 = 0;
                    while (ry < rows) : (ry += 1) {
                        const row_off = @as(usize, y_start + ry) * @as(usize, width) + x_start;
                        const pv: Vec16u8 = img.pixels[row_off..][0..16].*;
                        const diff: Vec16u8 = @max(pv, bg_v) - @min(pv, bg_v);
                        tile_fg[ry] = diff;
                        max_v = @max(max_v, diff);
                    }
                    const tile_max_diff: u32 = @reduce(.Max, max_v);
                    const thr: u8 = @intCast(@max(@as(u32, 36), (tile_max_diff * 30) / 100));
                    const thr_v: Vec16u8 = @splat(thr);

                    ry = 0;
                    while (ry < rows) : (ry += 1) {
                        const row_off = @as(usize, y_start + ry) * @as(usize, width) + x_start;
                        const diff = tile_fg[ry];
                        fg[row_off..][0..16].* = diff;
                        mask[row_off..][0..16].* = @select(u8, diff >= thr_v, ones16, zeros16);
                    }
                } else {
                    const bg_i: i32 = bg_u8;
                    var tile_max_diff: u32 = 0;
                    y = y_start;
                    while (y < y_end) : (y += 1) {
                        const row_off = @as(usize, y) * @as(usize, width);
                        var x = x_start;
                        while (x < x_end) : (x += 1) {
                            const p: i32 = img.pixels[row_off + x];
                            const diff: u32 = @abs(p - bg_i);
                            const d8: u8 = @intCast(@min(diff, 255));
                            fg[row_off + x] = d8;
                            if (d8 > tile_max_diff) tile_max_diff = d8;
                        }
                    }

                    const thr: u8 = @intCast(@max(@as(u32, 36), (tile_max_diff * 30) / 100));
                    y = y_start;
                    while (y < y_end) : (y += 1) {
                        const row_off = @as(usize, y) * @as(usize, width);
                        var x = x_start;
                        while (x < x_end) : (x += 1) {
                            mask[row_off + x] = if (fg[row_off + x] >= thr) 1 else 0;
                        }
                    }
                }
            }
        }
    }
}

/// Compute polarity-invariant foreground strength per pixel into pre-allocated fg/mask buffers (parallel SIMD).
fn computeForegroundInto(
    img: PgmImage,
    fg: []u8,
    mask: []u8,
) void {
    const n = @as(usize, img.width) * @as(usize, img.height);
    const tile: u32 = 16;
    const tw = (img.width + tile - 1) / tile;
    const th = (img.height + tile - 1) / tile;

    var ctx = FgWorkerCtx{
        .img = img,
        .fg = fg,
        .mask = mask,
        .tw = tw,
        .th = th,
        .next_ty = std.atomic.Value(u32).init(0),
    };

    const cpu_cnt = std.Thread.getCpuCount() catch 4;
    const n_threads: usize = if (n >= 262_144) @max(1, @min(@min(@as(usize, 8), cpu_cnt), (th + 3) / 4)) else 1;
    var threads: [8]?std.Thread = .{null} ** 8;
    if (n_threads > 1) {
        for (0..n_threads - 1) |ti| {
            threads[ti] = std.Thread.spawn(.{}, fgWorkerRun, .{&ctx}) catch null;
        }
    }
    fgWorkerRun(&ctx);
    for (threads) |maybe_t| {
        if (maybe_t) |t| t.join();
    }
}

/// Compute polarity-invariant foreground strength per pixel using 16x16 tile mode background (parallel SIMD).
fn computeForeground(
    gpa: std.mem.Allocator,
    img: PgmImage,
) !struct { fg: []u8, mask: []u8 } {
    const n = @as(usize, img.width) * @as(usize, img.height);
    const fg = try gpa.alloc(u8, n);
    errdefer gpa.free(fg);
    const mask = try gpa.alloc(u8, n);
    errdefer gpa.free(mask);
    computeForegroundInto(img, fg, mask);
    return .{ .fg = fg, .mask = mask };
}

const RgbToGrayCtx = struct {
    rgb: []const u8,
    gray: []u8,
    n_pixels: usize,
    next_px: std.atomic.Value(usize),
};

fn rgbToGrayWorker(ctx: *RgbToGrayCtx) void {
    const chunk: usize = 32768;
    const rgb = ctx.rgb;
    const gray = ctx.gray;
    const n = ctx.n_pixels;
    while (true) {
        const start = ctx.next_px.fetchAdd(chunk, .monotonic);
        if (start >= n) break;
        const end = @min(start + chunk, n);
        var i = start;
        while (i < end) : (i += 1) {
            const off = i * 3;
            const r: u32 = rgb[off];
            const g: u32 = rgb[off + 1];
            const b: u32 = rgb[off + 2];
            // ITU-R BT.601 luma matching PIL Image.convert("L"): (0.299 R + 0.587 G + 0.114 B)
            gray[i] = @intCast((r * 19595 + g * 38470 + b * 7471 + 32768) >> 16);
        }
    }
}

fn rgbToGrayParallel(rgb: []const u8, gray: []u8) void {
    const n = gray.len;
    var ctx = RgbToGrayCtx{
        .rgb = rgb,
        .gray = gray,
        .n_pixels = n,
        .next_px = std.atomic.Value(usize).init(0),
    };
    const cpu_cnt = std.Thread.getCpuCount() catch 4;
    const n_threads: usize = if (n >= 262_144) @max(1, @min(@as(usize, 8), cpu_cnt)) else 1;
    var threads: [8]?std.Thread = .{null} ** 8;
    if (n_threads > 1) {
        for (0..n_threads - 1) |ti| {
            threads[ti] = std.Thread.spawn(.{}, rgbToGrayWorker, .{&ctx}) catch null;
        }
    }
    rgbToGrayWorker(&ctx);
    for (threads) |maybe_t| {
        if (maybe_t) |t| t.join();
    }
}

const Coord16 = struct { x: u16, y: u16 };

/// Extract glyph-sized 8-connected components via bounded iterative BFS (in-place visited bit in mask, SIMD skip).
fn extractComponents(
    gpa: std.mem.Allocator,
    mask: []u8,
    width: u32,
    height: u32,
) !std.ArrayList(Comp) {
    var queue = try std.ArrayList(Coord16).initCapacity(gpa, 4096);
    defer queue.deinit(gpa);

    var comps = try std.ArrayList(Comp).initCapacity(gpa, 1024);
    errdefer comps.deinit(gpa);

    const dx = [_]i32{ -1, 0, 1, -1, 1, -1, 0, 1 };
    const dy = [_]i32{ -1, -1, -1, 0, 0, 1, 1, 1 };
    const Vec16u8 = @Vector(16, u8);
    const ones16: Vec16u8 = @splat(1);

    var y: u32 = 0;
    while (y < height) : (y += 1) {
        const row_off = @as(usize, y) * @as(usize, width);
        var x: u32 = 0;
        while (x < width) {
            // Fast SIMD skip over 16 background/already-visited pixels
            if (x + 16 <= width) {
                const v: Vec16u8 = mask[row_off + x ..][0..16].*;
                if (!@reduce(.Or, v == ones16)) {
                    x += 16;
                    continue;
                }
            }
            const idx = row_off + x;
            if (mask[idx] != 1) {
                x += 1;
                continue;
            }

            mask[idx] = 2;
            queue.clearRetainingCapacity();
            queue.appendAssumeCapacity(.{ .x = @intCast(x), .y = @intCast(y) });

            var min_x = x;
            var max_x = x;
            var min_y = y;
            var max_y = y;
            var area: u32 = 0;
            var head: usize = 0;
            var overflowed = false;

            while (head < queue.items.len) : (head += 1) {
                const cur = queue.items[head];
                const cx: u32 = cur.x;
                const cy_u: u32 = cur.y;
                area += 1;
                if (cx < min_x) min_x = cx;
                if (cx > max_x) max_x = cx;
                if (cy_u < min_y) min_y = cy_u;
                if (cy_u > max_y) max_y = cy_u;

                for (0..8) |dir| {
                    const nx_i = @as(i32, @intCast(cx)) + dx[dir];
                    const ny_i = @as(i32, @intCast(cy_u)) + dy[dir];
                    if (nx_i < 0 or ny_i < 0) continue;
                    const nx: u32 = @intCast(nx_i);
                    const ny: u32 = @intCast(ny_i);
                    if (nx >= width or ny >= height) continue;
                    const nidx = @as(usize, ny) * @as(usize, width) + nx;
                    if (mask[nidx] == 1) {
                        mask[nidx] = 2;
                        if (queue.items.len < 4096) {
                            queue.appendAssumeCapacity(.{ .x = @intCast(nx), .y = @intCast(ny) });
                        } else {
                            overflowed = true;
                        }
                    }
                }
            }

            x += 1;
            if (overflowed) continue;
            const bw = max_x - min_x + 1;
            const bh = max_y - min_y + 1;

            // Keep glyph-sized components, reject UI boxes/borders/lines.
            // Allow 1px-tall i-dots/dashes/underscores (., -, _) of width 2..14.
            if (bw == 0 or bh == 0) continue;
            if (bh == 1 and (bw < 2 or bw > 14)) continue;
            if (bw > 44 or bh > 40) continue;
            if (bw > 24 and bh <= 3) continue;
            if (bh > 24 and bw <= 3) continue;
            if (area < 2) continue;

            try comps.append(gpa, .{
                .x0 = min_x,
                .y0 = min_y,
                .x1 = max_x + 1,
                .y1 = max_y + 1,
                .area = area,
            });
        }
    }

    return comps;
}

fn countHolesGlyph(mask: []const u8, img_w: u32, x0: u32, y0: u32, x1: u32, y1: u32) u8 {
    const bw: usize = x1 - x0;
    const bh: usize = y1 - y0;
    if (bw == 0 or bh == 0 or bw > 48 or bh > 48) return 0;

    const pw = bw + 2;
    const ph = bh + 2;
    const plen = pw * ph;
    var padded: [50 * 50]u8 = undefined;
    var vis: [50 * 50]u8 = undefined;
    @memset(padded[0..plen], 0);
    @memset(vis[0..plen], 0);
    var qx: [50 * 50]u8 = undefined;
    var qy: [50 * 50]u8 = undefined;

    for (0..bh) |ry| {
        const row_off = @as(usize, y0 + @as(u32, @intCast(ry))) * @as(usize, img_w);
        for (0..bw) |rx| {
            padded[(ry + 1) * pw + (rx + 1)] = mask[row_off + x0 + @as(u32, @intCast(rx))];
        }
    }

    // Flood-fill exterior background from (0, 0)
    var head: usize = 0;
    var tail: usize = 0;
    vis[0] = 1;
    qx[tail] = 0;
    qy[tail] = 0;
    tail += 1;

    while (head < tail) : (head += 1) {
        const cx: usize = qx[head];
        const cy_u: usize = qy[head];
        const nbs = [_][2]isize{ .{ -1, 0 }, .{ 1, 0 }, .{ 0, -1 }, .{ 0, 1 } };
        for (nbs) |nb| {
            const nx_i = @as(isize, @intCast(cx)) + nb[0];
            const ny_i = @as(isize, @intCast(cy_u)) + nb[1];
            if (nx_i < 0 or ny_i < 0) continue;
            const nx: usize = @intCast(nx_i);
            const ny: usize = @intCast(ny_i);
            if (nx >= pw or ny >= ph) continue;
            const pidx = ny * pw + nx;
            if (vis[pidx] == 0 and padded[pidx] == 0) {
                vis[pidx] = 1;
                qx[tail] = @intCast(nx);
                qy[tail] = @intCast(ny);
                tail += 1;
            }
        }
    }

    var holes: u8 = 0;
    for (1..bh + 1) |ry| {
        for (1..bw + 1) |rx| {
            const pidx = ry * pw + rx;
            if (padded[pidx] == 0 and vis[pidx] == 0) {
                var h_area: u32 = 0;
                head = 0;
                tail = 0;
                vis[pidx] = 1;
                qx[tail] = @intCast(rx);
                qy[tail] = @intCast(ry);
                tail += 1;
                while (head < tail) : (head += 1) {
                    h_area += 1;
                    const cx: usize = qx[head];
                    const cy_u: usize = qy[head];
                    const nbs = [_][2]isize{ .{ -1, 0 }, .{ 1, 0 }, .{ 0, -1 }, .{ 0, 1 } };
                    for (nbs) |nb| {
                        const nx_i = @as(isize, @intCast(cx)) + nb[0];
                        const ny_i = @as(isize, @intCast(cy_u)) + nb[1];
                        if (nx_i < 0 or ny_i < 0) continue;
                        const nx: usize = @intCast(nx_i);
                        const ny: usize = @intCast(ny_i);
                        if (nx >= pw or ny >= ph) continue;
                        const nidx = ny * pw + nx;
                        if (vis[nidx] == 0 and padded[nidx] == 0) {
                            vis[nidx] = 1;
                            qx[tail] = @intCast(nx);
                            qy[tail] = @intCast(ny);
                            tail += 1;
                        }
                    }
                }
                if (h_area >= 2 and holes < 3) holes += 1;
            }
        }
    }
    return holes;
}

const QueryVecs = struct {
    fine: TemplateVec,
    coarse: Vec32i8,
};

/// Bilinear resample fg patch to 12x16, zero-mean unit-L2 normalize, and quantize to fine (6x32) + coarse (1x32) SIMD Vec32i8s.
fn extractGlyphVecs(
    fg: []const u8,
    img_w: u32,
    x0: u32,
    y0: u32,
    x1: u32,
    y1: u32,
) QueryVecs {
    const src_w: f32 = @floatFromInt(x1 - x0);
    const src_h: f32 = @floatFromInt(y1 - y0);
    const max_x: f32 = @max(0.0, src_w - 1.0);
    const max_y: f32 = @max(0.0, src_h - 1.0);

    var raw: [GRID_LEN]f32 = undefined;
    var sum: f32 = 0.0;

    // Precompute horizontal bilinear coords once (GRID_W=12) instead of 16x per glyph
    var ix0_lut: [GRID_W]u32 = undefined;
    var ix1_lut: [GRID_W]u32 = undefined;
    var fx_lut: [GRID_W]f32 = undefined;
    const bw_m1 = (x1 - x0) - 1;
    const scale_x = src_w / @as(f32, GRID_W);
    const scale_y = src_h / @as(f32, GRID_H);
    for (0..GRID_W) |gx| {
        const sx = std.math.clamp((@as(f32, @floatFromInt(gx)) + 0.5) * scale_x - 0.5, 0.0, max_x);
        const ix0: u32 = @intFromFloat(@floor(sx));
        ix0_lut[gx] = x0 + ix0;
        ix1_lut[gx] = x0 + @min(ix0 + 1, bw_m1);
        fx_lut[gx] = sx - @as(f32, @floatFromInt(ix0));
    }

    const bh_m1 = (y1 - y0) - 1;
    for (0..GRID_H) |gy| {
        const sy = std.math.clamp((@as(f32, @floatFromInt(gy)) + 0.5) * scale_y - 0.5, 0.0, max_y);
        const iy0: u32 = @intFromFloat(@floor(sy));
        const iy1: u32 = @min(iy0 + 1, bh_m1);
        const fy = sy - @as(f32, @floatFromInt(iy0));
        const one_m_fy = 1.0 - fy;
        const r0 = @as(usize, y0 + iy0) * @as(usize, img_w);
        const r1 = @as(usize, y0 + iy1) * @as(usize, img_w);

        for (0..GRID_W) |gx| {
            const c0 = ix0_lut[gx];
            const c1 = ix1_lut[gx];
            const fx = fx_lut[gx];
            const one_m_fx = 1.0 - fx;

            const v00: f32 = @floatFromInt(fg[r0 + c0]);
            const v01: f32 = @floatFromInt(fg[r0 + c1]);
            const v10: f32 = @floatFromInt(fg[r1 + c0]);
            const v11: f32 = @floatFromInt(fg[r1 + c1]);

            const top = v00 * one_m_fx + v01 * fx;
            const bot = v10 * one_m_fx + v11 * fx;
            const val = (top * one_m_fy + bot * fy) * (1.0 / 255.0);
            raw[gy * GRID_W + gx] = val;
            sum += val;
        }
    }

    const mean = sum / @as(f32, GRID_LEN);
    var sq_sum: f32 = 0.0;
    for (0..GRID_LEN) |i| {
        raw[i] -= mean;
        sq_sum += raw[i] * raw[i];
    }
    const inv_norm: f32 = if (sq_sum > 1e-10) 1.0 / @sqrt(sq_sum) else 0.0;
    var out_fine: TemplateVec = undefined;
    for (0..SIMD_I8_CHUNKS) |chunk| {
        var arr: [32]i8 = undefined;
        for (0..32) |k| {
            arr[k] = quantizeF32ToI8(raw[chunk * 32 + k] * inv_norm);
        }
        out_fine[chunk] = arr;
    }

    // 4x8 coarse descriptor (3x2 block average of 12x16 zero-mean grid)
    var coarse_f: [32]f32 = undefined;
    var c_sum: f32 = 0.0;
    for (0..8) |cy| {
        const r0 = (cy * 2) * GRID_W;
        const r1 = (cy * 2 + 1) * GRID_W;
        for (0..4) |cx| {
            const c0 = cx * 3;
            const v = (raw[r0 + c0] + raw[r0 + c0 + 1] + raw[r0 + c0 + 2] +
                raw[r1 + c0] + raw[r1 + c0 + 1] + raw[r1 + c0 + 2]) * (1.0 / 6.0);
            coarse_f[cy * 4 + cx] = v;
            c_sum += v;
        }
    }
    const c_mean = c_sum * (1.0 / 32.0);
    var c_sq: f32 = 0.0;
    for (0..32) |i| {
        coarse_f[i] -= c_mean;
        c_sq += coarse_f[i] * coarse_f[i];
    }
    const c_inv: f32 = if (c_sq > 1e-10) 1.0 / @sqrt(c_sq) else 0.0;
    var c_arr: [32]i8 = undefined;
    for (0..32) |i| {
        c_arr[i] = quantizeF32ToI8(coarse_f[i] * c_inv);
    }

    return .{ .fine = out_fine, .coarse = c_arr };
}

const MatchResult = struct {
    codepoint: u32,
    score: f32,
    second_cp: u32,
    second_score: f32,
};

const MatchedGlyph = struct {
    cp: u32,
    score: f32,
    second_cp: u32,
    second_score: f32,
    gap_before: f32,
    width: u32,
    height: u32,
    y0: u32 = 0,
    y1: u32 = 0,
    /// Set when case was decided from phrase-level x-height/cap-height geometry; later
    /// neighbor-based case heuristics must not override it.
    case_locked: bool = false,
};

inline fn dotI8Simd(q: *const TemplateVec, t: *const TemplateVec) f32 {
    // Accumulate pairs of chunks in i16 (max |2 * 127 * 127| = 32258 < 32767, overflow-free),
    // then widen to i32 only 3 times instead of 6.
    var acc: Vec32i32 = @splat(0);
    inline for (0..SIMD_I8_CHUNKS / 2) |pair| {
        const qa0: Vec32i16 = q[pair * 2];
        const tb0: Vec32i16 = t[pair * 2];
        const qa1: Vec32i16 = q[pair * 2 + 1];
        const tb1: Vec32i16 = t[pair * 2 + 1];
        const sum16: Vec32i16 = (qa0 * tb0) + (qa1 * tb1);
        acc += sum16;
    }
    const sum: i32 = @reduce(.Add, acc);
    return @as(f32, @floatFromInt(sum)) * INV_QUANT_SQ;
}

inline fn dotCoarseI8Simd(q: Vec32i8, t: Vec32i8) f32 {
    const qa: Vec32i16 = q;
    const tb: Vec32i16 = t;
    const prod: Vec32i16 = qa * tb;
    const acc: Vec32i32 = prod;
    const sum: i32 = @reduce(.Add, acc);
    return @as(f32, @floatFromInt(sum)) * INV_QUANT_SQ;
}

/// Hot-path SoA i8 SIMD shape matcher returning top-2 distinct codepoints (aspect-indexed + 4x8 coarse filter).
fn matchGlyphSimd(
    atlas: Atlas,
    q: *const QueryVecs,
    q_aspect: f32,
    q_y_top: f32,
    q_y_bot: f32,
    q_holes: u8,
) MatchResult {
    const meta = atlas.meta;
    const coarse = atlas.coarse;
    const vecs = atlas.vecs;
    const use_coarse = atlas.has_coarse;
    const q_coarse = q.coarse;
    const q_vecs = &q.fine;
    const aspect_win: f32 = 0.35;

    // Binary search for center template near q_aspect so best_score/second_score warm up immediately
    // and coarse/upper-bound pruning rejects distant templates early.
    var lo: usize = 0;
    var hi: usize = meta.len;
    while (lo < hi) {
        const mid = lo + (hi - lo) / 2;
        if (meta[mid].aspect < q_aspect) {
            lo = mid + 1;
        } else {
            hi = mid;
        }
    }
    const center = lo;

    var best_cp: u32 = '?';
    var best_score: f32 = -10.0;
    var second_cp: u32 = '?';
    var second_score: f32 = -10.0;

    // Scan upward from center to max_aspect
    var idx = center;
    while (idx < meta.len) : (idx += 1) {
        const m = meta[idx];
        const aspect_diff = m.aspect - q_aspect;
        if (aspect_diff > aspect_win) break;
        if (1.05 - 0.22 * aspect_diff <= second_score and 1.05 - 0.22 * aspect_diff <= best_score) break;

        const y_diff = (@abs(q_y_top - m.y_top) + @abs(q_y_bot - m.y_bot)) * 0.5;
        if (y_diff > 0.85) continue;

        const hole_pen: f32 = if (q_holes == m.holes) 0.0 else 0.10;
        const penalty = 0.22 * aspect_diff + 0.28 * @min(y_diff, 1.5) + hole_pen;
        const cutoff = if (m.codepoint == best_cp) best_score else second_score;
        if (1.05 - penalty <= cutoff) continue;

        if (use_coarse and cutoff > 0.35) {
            const cdot = dotCoarseI8Simd(q_coarse, coarse[idx]);
            if (cdot - penalty < cutoff - 0.16) continue;
        }

        const dot = dotI8Simd(q_vecs, &vecs[idx]);
        const score = dot - penalty;
        if (score > best_score) {
            if (m.codepoint != best_cp) {
                second_score = best_score;
                second_cp = best_cp;
            }
            best_score = score;
            best_cp = m.codepoint;
        } else if (m.codepoint != best_cp and score > second_score) {
            second_score = score;
            second_cp = m.codepoint;
        }
    }

    // Scan downward from center - 1 to min_aspect
    var rev = center;
    while (rev > 0) {
        rev -= 1;
        const m = meta[rev];
        const aspect_diff = q_aspect - m.aspect;
        if (aspect_diff > aspect_win) break;
        if (1.05 - 0.22 * aspect_diff <= second_score and 1.05 - 0.22 * aspect_diff <= best_score) break;

        const y_diff = (@abs(q_y_top - m.y_top) + @abs(q_y_bot - m.y_bot)) * 0.5;
        if (y_diff > 0.85) continue;

        const hole_pen: f32 = if (q_holes == m.holes) 0.0 else 0.10;
        const penalty = 0.22 * aspect_diff + 0.28 * @min(y_diff, 1.5) + hole_pen;
        const cutoff = if (m.codepoint == best_cp) best_score else second_score;
        if (1.05 - penalty <= cutoff) continue;

        if (use_coarse and cutoff > 0.35) {
            const cdot = dotCoarseI8Simd(q_coarse, coarse[rev]);
            if (cdot - penalty < cutoff - 0.16) continue;
        }

        const dot = dotI8Simd(q_vecs, &vecs[rev]);
        const score = dot - penalty;
        if (score > best_score) {
            if (m.codepoint != best_cp) {
                second_score = best_score;
                second_cp = best_cp;
            }
            best_score = score;
            best_cp = m.codepoint;
        } else if (m.codepoint != best_cp and score > second_score) {
            second_score = score;
            second_cp = m.codepoint;
        }
    }

    // Rare fallback for extreme aspect ratios (e.g. wide dashes)
    if (best_score <= -9.0) {
        for (meta, 0..) |m, fi| {
            const aspect_diff = @abs(q_aspect - m.aspect);
            if (aspect_diff > 1.6) continue;
            const dot = dotI8Simd(q_vecs, &vecs[fi]);
            const y_diff = (@abs(q_y_top - m.y_top) + @abs(q_y_bot - m.y_bot)) * 0.5;
            const hole_pen: f32 = if (q_holes == m.holes) 0.0 else 0.10;
            const score = dot - 0.22 * aspect_diff - 0.28 * @min(y_diff, 1.5) - hole_pen;
            if (score > best_score) {
                if (m.codepoint != best_cp) {
                    second_score = best_score;
                    second_cp = best_cp;
                }
                best_score = score;
                best_cp = m.codepoint;
            } else if (m.codepoint != best_cp and score > second_score) {
                second_score = score;
                second_cp = m.codepoint;
            }
        }
    }

    return .{
        .codepoint = best_cp,
        .score = best_score,
        .second_cp = second_cp,
        .second_score = second_score,
    };
}

fn trimVerticalBounds(fg: []const u8, img_w: u32, x0: u32, y0: u32, x1: u32, y1: u32) struct { y0: u32, y1: u32, ok: bool } {
    var ny0: u32 = y1;
    var ny1: u32 = y0;
    var y = y0;
    while (y < y1) : (y += 1) {
        const row_off = @as(usize, y) * @as(usize, img_w);
        var x = x0;
        while (x < x1) : (x += 1) {
            if (fg[row_off + x] >= 36) {
                if (y < ny0) ny0 = y;
                if (y + 1 > ny1) ny1 = y + 1;
                break;
            }
        }
    }
    if (ny1 <= ny0) return .{ .y0 = y0, .y1 = y1, .ok = false };
    return .{ .y0 = ny0, .y1 = ny1, .ok = true };
}

fn matchBox(
    fg: []const u8,
    mask: []const u8,
    img_w: u32,
    atlas: Atlas,
    x0: u32,
    y0: u32,
    x1: u32,
    y1: u32,
    ref_top: f32,
    ref_h: f32,
) MatchResult {
    const bw = x1 - x0;
    const bh = y1 - y0;
    const q_vecs = extractGlyphVecs(fg, img_w, x0, y0, x1, y1);
    const q_aspect = @as(f32, @floatFromInt(bw)) / @as(f32, @floatFromInt(bh));
    const q_y_top = (@as(f32, @floatFromInt(y0)) - ref_top) / ref_h;
    const q_y_bot = (@as(f32, @floatFromInt(y1)) - ref_top) / ref_h;
    const q_holes = countHolesGlyph(mask, img_w, x0, y0, x1, y1);
    return matchGlyphSimd(atlas, &q_vecs, q_aspect, q_y_top, q_y_bot, q_holes);
}

/// Match a component, splitting touching glyphs along vertical ink valleys when two halves score better.
fn matchOrSplitGlyph(
    gpa: std.mem.Allocator,
    out: *std.ArrayList(MatchedGlyph),
    fg: []const u8,
    mask: []const u8,
    img_w: u32,
    atlas: Atlas,
    x0: u32,
    y0: u32,
    x1: u32,
    y1: u32,
    gap_before: f32,
    ref_top: f32,
    ref_h: f32,
    depth: u8,
) !void {
    const bw = x1 - x0;
    const bh = y1 - y0;
    const single = matchBox(fg, mask, img_w, atlas, x0, y0, x1, y1, ref_top, ref_h);
    const q_aspect = @as(f32, @floatFromInt(bw)) / @as(f32, @floatFromInt(bh));

    const is_wide_single = (single.codepoint == 'm' or single.codepoint == 'w' or single.codepoint == 'M' or single.codepoint == 'W');
    const can_try_split = (depth < 2) and (bw >= 10) and (bh >= 5) and (bw <= 48) and (q_aspect >= 1.00) and
        ((!is_wide_single and single.score < 0.80) or (is_wide_single and single.score < 0.72));

    if (can_try_split) {
        var col_sum: [48]u32 = .{0} ** 48;
        var x: u32 = 0;
        while (x < bw) : (x += 1) {
            var s: u32 = 0;
            var y = y0;
            while (y < y1) : (y += 1) {
                s += fg[@as(usize, y) * @as(usize, img_w) + (x0 + x)];
            }
            col_sum[x] = s;
        }

        // Collect up to 2 deepest local-minimum valleys (separated by >= 3px) and pick the highest-scoring split
        var cand_sx: [2]u32 = .{ 0, 0 };
        var cand_val: [2]u32 = .{ std.math.maxInt(u32), std.math.maxInt(u32) };
        var cand_cnt: usize = 0;

        var sx: u32 = 3;
        while (sx + 3 <= bw) : (sx += 1) {
            var lmax: u32 = 0;
            for (0..sx) |lx| {
                if (col_sum[lx] > lmax) lmax = col_sum[lx];
            }
            var rmax: u32 = 0;
            for (sx + 1..bw) |rx| {
                if (col_sum[rx] > rmax) rmax = col_sum[rx];
            }
            const side_min = @min(lmax, rmax);
            if (side_min < 120) continue;
            if (col_sum[sx] * 10 > side_min * 7) continue; // not a real valley
            // Prefer local minima
            if (col_sum[sx] > col_sum[sx - 1] or col_sum[sx] > col_sum[sx + 1]) continue;

            // Merge with an existing nearby candidate (< 3px away) if deeper
            var replaced_near = false;
            for (0..cand_cnt) |ci| {
                const dist = if (sx > cand_sx[ci]) sx - cand_sx[ci] else cand_sx[ci] - sx;
                if (dist < 3) {
                    if (col_sum[sx] < cand_val[ci]) {
                        cand_sx[ci] = sx;
                        cand_val[ci] = col_sum[sx];
                    }
                    replaced_near = true;
                    break;
                }
            }
            if (replaced_near) continue;

            if (cand_cnt < 2) {
                cand_sx[cand_cnt] = sx;
                cand_val[cand_cnt] = col_sum[sx];
                cand_cnt += 1;
            } else {
                const worst_i: usize = if (cand_val[0] >= cand_val[1]) 0 else 1;
                if (col_sum[sx] < cand_val[worst_i]) {
                    cand_sx[worst_i] = sx;
                    cand_val[worst_i] = col_sum[sx];
                }
            }
        }

        var best_split_x: u32 = 0;
        var best_split_mean: f32 = -10.0;
        var best_lb_y0: u32 = y0;
        var best_lb_y1: u32 = y1;
        var best_rb_y0: u32 = y0;
        var best_rb_y1: u32 = y1;

        for (0..cand_cnt) |ci| {
            const split_x = x0 + cand_sx[ci];
            const lb = trimVerticalBounds(fg, img_w, x0, y0, split_x, y1);
            const rb = trimVerticalBounds(fg, img_w, split_x, y0, x1, y1);
            if (lb.ok and rb.ok and (lb.y1 - lb.y0) >= 2 and (rb.y1 - rb.y0) >= 2) {
                const lm = matchBox(fg, mask, img_w, atlas, x0, lb.y0, split_x, lb.y1, ref_top, ref_h);
                const rm = matchBox(fg, mask, img_w, atlas, split_x, rb.y0, x1, rb.y1, ref_top, ref_h);
                const mean_lr = (lm.score + rm.score) * 0.5;
                const min_lr = @min(lm.score, rm.score);
                const margin: f32 = if (is_wide_single) 0.07 else 0.025;
                const accept = (min_lr >= 0.42 and mean_lr > single.score + margin) or
                    (single.score < 0.62 and min_lr >= 0.45 and mean_lr > single.score - 0.02);
                if (accept and mean_lr > best_split_mean) {
                    best_split_mean = mean_lr;
                    best_split_x = split_x;
                    best_lb_y0 = lb.y0;
                    best_lb_y1 = lb.y1;
                    best_rb_y0 = rb.y0;
                    best_rb_y1 = rb.y1;
                }
            }
        }

        if (best_split_x > x0) {
            try matchOrSplitGlyph(gpa, out, fg, mask, img_w, atlas, x0, best_lb_y0, best_split_x, best_lb_y1, gap_before, ref_top, ref_h, depth + 1);
            try matchOrSplitGlyph(gpa, out, fg, mask, img_w, atlas, best_split_x, best_rb_y0, x1, best_rb_y1, 0.0, ref_top, ref_h, depth + 1);
            return;
        }
    }

    try out.append(gpa, .{
        .cp = single.codepoint,
        .score = single.score,
        .second_cp = single.second_cp,
        .second_score = single.second_score,
        .gap_before = gap_before,
        .width = bw,
        .height = bh,
        .y0 = y0,
        .y1 = y1,
    });
}

fn isAsciiDigit(cp: u32) bool {
    return cp >= '0' and cp <= '9';
}

fn isAsciiLower(cp: u32) bool {
    return (cp >= 'a' and cp <= 'z') or cp == 0x00E1 or cp == 0x00E9 or cp == 0x00ED or cp == 0x00F3 or cp == 0x00FA or cp == 0x00F1 or cp == 0x00FC;
}

fn isAsciiUpper(cp: u32) bool {
    return (cp >= 'A' and cp <= 'Z') or cp == 0x00C1 or cp == 0x00C9 or cp == 0x00CD or cp == 0x00D3 or cp == 0x00DA or cp == 0x00D1;
}

fn isSameShapeUpper(cp: u32) bool {
    return cp == 'C' or cp == 'O' or cp == 'S' or cp == 'V' or cp == 'W' or cp == 'X' or cp == 'Z' or cp == 'U' or cp == 'P' or cp == 'K';
}

fn isSameShapeLower(cp: u32) bool {
    return cp == 'c' or cp == 'o' or cp == 's' or cp == 'v' or cp == 'w' or cp == 'x' or cp == 'z' or cp == 'u' or cp == 'p' or cp == 'k';
}

fn isAsciiAlpha(cp: u32) bool {
    return isAsciiLower(cp) or isAsciiUpper(cp);
}

/// Letters whose upper and lower case share one normalized shape and differ only in height
/// (both sit on the baseline, no ascender/descender difference). P/p and K/k are excluded:
/// they differ by descender/ascender, not by height.
fn isHeightOnlyCasePair(cp: u32) bool {
    return switch (cp) {
        'c', 'o', 's', 'v', 'w', 'x', 'z', 'u', 'C', 'O', 'S', 'V', 'W', 'X', 'Z', 'U' => true,
        else => false,
    };
}

fn medianSmall(vals: []u32) f32 {
    std.mem.sort(u32, vals, {}, u32Less);
    const m = vals.len / 2;
    if (vals.len % 2 == 1) return @floatFromInt(vals[m]);
    return (@as(f32, @floatFromInt(vals[m - 1])) + @as(f32, @floatFromInt(vals[m]))) * 0.5;
}

/// Decide the case of height-only case pairs (C/c, O/o, S/s, ...) from the phrase's own
/// x-height (a e m n r) and cap height (distinct capitals, digits). Ascenders (b d h l) are NOT
/// cap references: at small sizes they are often 1-2 px taller than capitals.
/// The atlas normalizes glyph boxes, so the matcher cannot tell these apart by shape.
/// Without an x-height reference nothing is decided (the neighbor heuristics still apply).
fn resolveSameShapeCase(glyphs: []MatchedGlyph) void {
    var xs: [64]u32 = undefined;
    var cs: [64]u32 = undefined;
    var nx: usize = 0;
    var nc: usize = 0;
    for (glyphs) |g| {
        switch (g.cp) {
            'a', 'e', 'm', 'n', 'r' => if (nx < xs.len) {
                xs[nx] = g.height;
                nx += 1;
            },
            'A', 'B', 'D', 'E', 'F', 'G', 'H', 'L', 'M', 'N', 'R', 'T', '2', '3', '4', '5', '6', '7', '9' => if (nc < cs.len) {
                cs[nc] = g.height;
                nc += 1;
            },
            else => {},
        }
    }
    if (nx == 0) return;
    const xh = medianSmall(xs[0..nx]);
    // Typical sans/mono cap-height / x-height ratio is ~1.33-1.40; used only when no capital is present.
    const ch = if (nc > 0) medianSmall(cs[0..nc]) else xh * 1.37;
    if (ch < xh + 1.5) return; // not separable at this size (or references are wrong)
    const mid = (xh + ch) * 0.5;
    for (glyphs) |*g| {
        if (!isHeightOnlyCasePair(g.cp)) continue;
        // Round letters overshoot both lines by ~1px; the x refs (a e) partly overshoot too.
        const round = switch (g.cp) {
            'c', 'o', 's', 'C', 'O', 'S' => true,
            else => false,
        };
        const thr = if (round) mid + 0.5 else mid;
        const h: f32 = @floatFromInt(g.height);
        const lower = h <= thr;
        const is_up = g.cp >= 'A' and g.cp <= 'Z';
        if (lower and is_up) g.cp += 32;
        if (!lower and !is_up) g.cp -= 32;
        g.case_locked = true;
    }
}

/// Disambiguate isolated-glyph confusions (¡/i, 8/g, 1/i/l/I, 0/O/o, í/i, I/l/i, and same-shape case)
/// using immediate token neighbors and baseline/cap-height geometry.
fn disambiguatePhrase(glyphs: []MatchedGlyph, space_gap: f32, ref_top: f32, ref_bot: f32) void {
    const n = glyphs.len;
    if (n < 2) return;
    resolveSameShapeCase(glyphs);

    for (0..n) |i| {
        const g = &glyphs[i];
        const has_left = (i > 0) and (g.gap_before < space_gap * 1.15);
        const has_right = (i + 1 < n) and (glyphs[i + 1].gap_before < space_gap * 1.15);
        const left_cp: u32 = if (has_left) glyphs[i - 1].cp else 0;
        const right_cp: u32 = if (has_right) glyphs[i + 1].cp else 0;

        const near_digit = isAsciiDigit(left_cp) or isAsciiDigit(right_cp);
        const near_lower = isAsciiLower(left_cp) or isAsciiLower(right_cp);
        const near_upper = isAsciiUpper(left_cp) or isAsciiUpper(right_cp);
        const near_alpha = near_lower or near_upper;

        // 0a) '¡' (inverted bang) inside a word (e.g., real¡zada, Evidenc¡a, P¡de) or non-descending before lowercase (¡magen) -> 'i'
        //     Preserves real '¡Hola!' (word-start + uppercase/descending).
        if (g.cp == 0x00A1) {
            if (has_left and isAsciiAlpha(left_cp)) {
                g.cp = 'i';
                continue;
            }
            if (!has_left and has_right and isAsciiLower(right_cp) and @as(f32, @floatFromInt(g.y1)) <= ref_bot + 1.0) {
                g.cp = 'i';
                continue;
            }
        }

        // 0b) Weak 'í' vs 'i' confusion on mono fonts where i-dot slightly resembles acute accent
        if (g.cp == 0x00ED and g.second_cp == 'i' and (g.score - g.second_score) < 0.035) {
            g.cp = 'i';
            continue;
        }

        // 0c) '8' inside a purely alphabetic lowercase word (e.g., FlowSi8ht, ima8en, pre8unta, 8lobal) -> 'g'
        //     Double-story sans-serif 'g' shares holes=2 with '8', but descends below baseline or starts at x-height.
        if (g.cp == '8' and near_lower and !near_digit) {
            const looks_like_g = (g.second_cp == 'g') or
                (@as(f32, @floatFromInt(g.y1)) > ref_bot + 1.0) or
                (@as(f32, @floatFromInt(g.y0)) >= ref_top + 1.5);
            if (looks_like_g) {
                g.cp = 'g';
                continue;
            }
        }

        // 1) 'O' or 'o' inside a numeric token (e.g., 2O26, 11:O1) -> '0'
        if ((g.cp == 'O' or g.cp == 'o') and (near_digit or ((left_cp == ':' or right_cp == ':') and !near_alpha))) {
            if (g.second_cp == '0' or (g.score - g.second_score < 0.25)) {
                g.cp = '0';
            }
            continue;
        }

        // 2) '0' inside a purely alphabetic word (e.g., C0DER, H0la) -> 'O' or 'o'
        if (g.cp == '0' and near_alpha and !near_digit) {
            if (g.second_cp == 'O' or g.second_cp == 'o') {
                g.cp = g.second_cp;
            } else {
                g.cp = if (near_lower) 'o' else 'O';
            }
            continue;
        }

        // 3) '1' inside a lowercase alphabetic word (e.g., sal1da, d1spon1ble, 1sycode) -> 'i' or 'l'
        if (g.cp == '1' and near_lower and !near_digit) {
            if ((g.second_cp == 'i' or g.second_cp == 'l' or g.second_cp == 'I') and (g.score - g.second_score) <= 0.18) {
                g.cp = g.second_cp;
            } else {
                g.cp = 'i';
            }
            continue;
        }

        // 4) 'l' or 'I' or '|' adjacent to digits without letters -> '1'
        if ((g.cp == 'l' or g.cp == 'I' or g.cp == '|') and near_digit and !near_alpha) {
            if (g.second_cp == '1' and (g.score - g.second_score) <= 0.20) {
                g.cp = '1';
            }
            continue;
        }

        // 4b) Sans-serif 'I' inside lowercase words (e.g., pipeIine, soIo, eI, shouId, FIow, detaIIe, Ilamada, artIfacts) -> 'l' or 'i'
        if (g.cp == 'I' and !near_digit) {
            const after_lower = has_left and isAsciiLower(left_cp);
            const before_distinct_upper = has_right and isAsciiUpper(right_cp) and right_cp != 'I';
            const after_upper_before_lower = has_left and isAsciiUpper(left_cp) and has_right and isAsciiLower(right_cp);
            const double_l_start = !has_left and has_right and (right_cp == 'l' or right_cp == 'I') and
                (i + 2 < n and glyphs[i + 2].gap_before < space_gap * 1.15 and isAsciiLower(glyphs[i + 2].cp));
            if ((after_lower and !before_distinct_upper) or after_upper_before_lower or double_l_start) {
                if (g.second_cp == 'i' and (g.score - g.second_score) <= 0.12) {
                    g.cp = 'i';
                } else {
                    g.cp = 'l';
                }
                continue;
            }
        }

        // 5) Same-shape uppercase inside a lowercase word (e.g., DeSCribe, BuSCar, archiVOs, ComandoS)
        //    Only lowercases when NOT at word start, OR when at word start with same x-height as right lowercase neighbor.
        if (isSameShapeUpper(g.cp) and near_lower and !g.case_locked) {
            const left_is_distinct_upper = has_left and isAsciiUpper(left_cp) and !isSameShapeUpper(left_cp);
            const right_is_distinct_upper = has_right and isAsciiUpper(right_cp) and !isSameShapeUpper(right_cp);
            if (!left_is_distinct_upper and !right_is_distinct_upper) {
                if (has_left and isAsciiLower(left_cp)) {
                    const lh = glyphs[i - 1].height;
                    if (g.height <= lh + 1) {
                        g.cp = g.cp + 32;
                        continue;
                    }
                }
                if (has_left and has_right and isAsciiLower(right_cp)) {
                    const rh = glyphs[i + 1].height;
                    if (g.height <= rh + 1) {
                        g.cp = g.cp + 32;
                        continue;
                    }
                }
            }
        }

        // 5b) ':' inside a lowercase word (e.g. Conf:gurac:ón -> Configuración when i-dot gap is wide)
        if (g.cp == ':' and has_left and has_right and isAsciiLower(left_cp) and (isAsciiLower(right_cp) or isAsciiDigit(right_cp))) {
            const ref_h_col: f32 = @max(6.0, ref_bot - ref_top);
            const tall_like_i = ref_h_col >= 7.0 and @as(f32, @floatFromInt(g.height)) >= ref_h_col * 0.74;
            const second_is_i = (g.second_cp == 'i' and (g.score - g.second_score) <= 0.16);
            if (second_is_i or (tall_like_i and g.width <= 4)) {
                g.cp = 'i';
                continue;
            }
        }

        // 6) Same-shape lowercase inside an ALL-CAPS token (e.g., FL0WsIGHT -> FLOWSIGHT)
        //    Only uppercases when height matches the uppercase neighbors (never uppercases x-height 'u' in 'BuSCar').
        if (isSameShapeLower(g.cp) and !g.case_locked and has_left and has_right and isAsciiUpper(left_cp) and isAsciiUpper(right_cp)) {
            if (g.height + 1 >= glyphs[i - 1].height and g.height + 1 >= glyphs[i + 1].height) {
                g.cp = g.cp - 32;
                continue;
            }
        }
    }
}

fn isTightRightPunct(cp: u32) bool {
    return cp == '.' or cp == ',' or cp == ':' or cp == ';' or cp == '!' or cp == '?' or cp == ')' or cp == ']';
}

fn isTightLeftPunct(cp: u32) bool {
    return cp == 0x00A1 or cp == 0x00BF or cp == '(' or cp == '[' or cp == '/';
}

fn compLessByY(_: void, a: Comp, b: Comp) bool {
    if (a.y0 != b.y0) return a.y0 < b.y0;
    return a.x0 < b.x0;
}

fn compLessByX(_: void, a: Comp, b: Comp) bool {
    if (a.x0 != b.x0) return a.x0 < b.x0;
    return a.y0 < b.y0;
}

fn u32Less(_: void, a: u32, b: u32) bool {
    return a < b;
}

const Band = struct {
    x0: u32,
    x1: u32,
    y0: u32,
    y1: u32,
    core_sum_cy: f32,
    core_sum_y1: f32,
    core_sum_h: f32,
    core_cnt: u32,
    merged: bool = false,
    indices: std.ArrayList(usize),
};

const BandWorkerCtx = struct {
    gpa: std.mem.Allocator,
    img_w: u32,
    fg: []const u8,
    mask: []const u8,
    atlas: Atlas,
    comps: []const Comp,
    bands: []const Band,
    band_order: []const usize,
    band_outs: []std.ArrayList(OcrLine),
    next_task: std.atomic.Value(usize),
};

fn bandWorkerRun(ctx: *BandWorkerCtx) void {
    const gpa = ctx.gpa;
    var band_comps = std.ArrayList(Comp).initCapacity(gpa, 128) catch return;
    defer band_comps.deinit(gpa);
    var merged_glyphs = std.ArrayList(Comp).initCapacity(gpa, 128) catch return;
    defer merged_glyphs.deinit(gpa);
    var phrase_buf = std.ArrayList(u8).initCapacity(gpa, 256) catch return;
    defer phrase_buf.deinit(gpa);
    var tmp_u32 = std.ArrayList(u32).initCapacity(gpa, 128) catch return;
    defer tmp_u32.deinit(gpa);
    var tops = std.ArrayList(u32).initCapacity(gpa, 128) catch return;
    defer tops.deinit(gpa);
    var bots = std.ArrayList(u32).initCapacity(gpa, 128) catch return;
    defer bots.deinit(gpa);
    var matched = std.ArrayList(MatchedGlyph).initCapacity(gpa, 128) catch return;
    defer matched.deinit(gpa);

    while (true) {
        const ti = ctx.next_task.fetchAdd(1, .monotonic);
        if (ti >= ctx.band_order.len) break;
        const bi = ctx.band_order[ti];
        const b = ctx.bands[bi];
        if (b.merged or b.indices.items.len == 0) continue;

        band_comps.clearRetainingCapacity();
        for (b.indices.items) |ci| {
            band_comps.append(gpa, ctx.comps[ci]) catch return;
        }
        std.mem.sort(Comp, band_comps.items, {}, compLessByX);

        // Merge vertically aligned diacritics / dots (i, j, á, ñ, :, ;, ¡, ¿)
        merged_glyphs.clearRetainingCapacity();
        for (band_comps.items) |c| {
            var did_merge = false;
            var mi: usize = merged_glyphs.items.len;
            while (mi > 0) {
                mi -= 1;
                const prev = &merged_glyphs.items[mi];
                if (c.x0 > prev.x1 + 2) break;
                const ov_x0 = @max(c.x0, prev.x0);
                const ov_x1 = @min(c.x1, prev.x1);
                const cx_c = (@as(i32, @intCast(c.x0)) + @as(i32, @intCast(c.x1))) >> 1;
                const cx_p = (@as(i32, @intCast(prev.x0)) + @as(i32, @intCast(prev.x1))) >> 1;
                const horiz_aligned = (ov_x1 > ov_x0) or (@abs(cx_c - cx_p) <= 2);
                const nx0 = @min(c.x0, prev.x0);
                const ny0 = @min(c.y0, prev.y0);
                const nx1 = @max(c.x1, prev.x1);
                const ny1 = @max(c.y1, prev.y1);
                const v_overlap = @min(c.y1, prev.y1) > @max(c.y0, prev.y0) + 1;
                const v_gap: u32 = if (c.y0 > prev.y1) c.y0 - prev.y1 else if (prev.y0 > c.y1) prev.y0 - c.y1 else 0;
                if (!v_overlap and horiz_aligned and v_gap <= 7 and (ny1 - ny0) <= 30 and (nx1 - nx0) <= 22) {
                    prev.x0 = nx0;
                    prev.y0 = ny0;
                    prev.x1 = nx1;
                    prev.y1 = ny1;
                    prev.area += c.area;
                    did_merge = true;
                    break;
                }
            }
            if (!did_merge) {
                merged_glyphs.append(gpa, c) catch return;
            }
        }

        if (merged_glyphs.items.len == 0) continue;

        tmp_u32.clearRetainingCapacity();
        for (merged_glyphs.items) |g| {
            if (g.height() >= 5) tmp_u32.append(gpa, g.height()) catch return;
        }
        const band_med_h: u32 = if (tmp_u32.items.len > 0) blk: {
            std.mem.sort(u32, tmp_u32.items, {}, u32Less);
            break :blk tmp_u32.items[tmp_u32.items.len / 2];
        } else 12;

        const phrase_split_gap: u32 = @max(22, (band_med_h * 22) / 10);

        var p_start: usize = 0;
        while (p_start < merged_glyphs.items.len) {
            var p_end = p_start + 1;
            while (p_end < merged_glyphs.items.len) : (p_end += 1) {
                const prev = merged_glyphs.items[p_end - 1];
                const cur = merged_glyphs.items[p_end];
                const gap = if (cur.x0 > prev.x1) cur.x0 - prev.x1 else 0;
                if (gap > phrase_split_gap) break;
            }

            const slice = merged_glyphs.items[p_start..p_end];
            p_start = p_end;

            tmp_u32.clearRetainingCapacity();
            tops.clearRetainingCapacity();
            bots.clearRetainingCapacity();

            var px0: u32 = slice[0].x0;
            var py0: u32 = slice[0].y0;
            var px1: u32 = slice[0].x1;
            var py1: u32 = slice[0].y1;

            for (slice) |g| {
                if (g.x0 < px0) px0 = g.x0;
                if (g.y0 < py0) py0 = g.y0;
                if (g.x1 > px1) px1 = g.x1;
                if (g.y1 > py1) py1 = g.y1;
                if (g.height() >= 5) {
                    tops.append(gpa, g.y0) catch return;
                    bots.append(gpa, g.y1) catch return;
                    tmp_u32.append(gpa, g.width()) catch return;
                }
            }
            if (tops.items.len == 0) continue;

            std.mem.sort(u32, tops.items, {}, u32Less);
            std.mem.sort(u32, bots.items, {}, u32Less);
            std.mem.sort(u32, tmp_u32.items, {}, u32Less);

            const ref_top: f32 = @floatFromInt(tops.items[tops.items.len / 10]);
            const ref_bot: f32 = @floatFromInt(bots.items[(bots.items.len * 3) / 5]);
            const ref_h: f32 = @max(6.0, ref_bot - ref_top);
            const med_w: f32 = @floatFromInt(tmp_u32.items[tmp_u32.items.len / 2]);
            const p25_w: f32 = @floatFromInt(tmp_u32.items[tmp_u32.items.len / 4]);
            const is_monospace_like = (tmp_u32.items.len >= 6 and p25_w >= med_w * 0.78);
            var space_gap: f32 = @max(3.2, @min(med_w * 0.52, ref_h * 0.36));

            // Monospace adaptive spacing: only when glyph widths are uniform (fixed-pitch cells
            // around narrow letters like r/t/i) scale space threshold above intra-word padding.
            if (is_monospace_like) {
                tmp_u32.clearRetainingCapacity();
                for (1..slice.len) |gi| {
                    if (slice[gi].x0 > slice[gi - 1].x1) {
                        const gpx = slice[gi].x0 - slice[gi - 1].x1;
                        if (gpx > 0 and gpx <= 12) tmp_u32.append(gpa, gpx) catch return;
                    }
                }
                if (tmp_u32.items.len >= 4) {
                    std.mem.sort(u32, tmp_u32.items, {}, u32Less);
                    const med_gap: f32 = @floatFromInt(tmp_u32.items[tmp_u32.items.len / 2]);
                    if (med_gap >= 2.0) {
                        space_gap = @max(space_gap, med_gap * 2.15);
                    }
                }
            }

            phrase_buf.clearRetainingCapacity();
            matched.clearRetainingCapacity();

            const allow_split: u8 = if (slice.len >= 2) 0 else 2;
            for (slice, 0..) |g, gi| {
                const gap: f32 = if (gi > 0 and g.x0 > slice[gi - 1].x1)
                    @floatFromInt(g.x0 - slice[gi - 1].x1)
                else
                    0.0;

                matchOrSplitGlyph(
                    gpa,
                    &matched,
                    ctx.fg,
                    ctx.mask,
                    ctx.img_w,
                    ctx.atlas,
                    g.x0,
                    g.y0,
                    g.x1,
                    g.y1,
                    gap,
                    ref_top,
                    ref_h,
                    allow_split,
                ) catch return;
            }

            const char_cnt: u32 = @intCast(matched.items.len);
            if (char_cnt == 0) continue;

            var score_sum: f32 = 0.0;
            var min_score: f32 = 1.0;
            for (matched.items) |mg| {
                const clamped = std.math.clamp(mg.score, 0.0, 1.0);
                score_sum += clamped;
                if (clamped < min_score) min_score = clamped;
            }

            // Contextual disambiguation (2 passes so cascaded pairs resolve cleanly)
            disambiguatePhrase(matched.items, space_gap, ref_top, ref_bot);
            disambiguatePhrase(matched.items, space_gap, ref_top, ref_bot);

            for (matched.items, 0..) |mg, gi| {
                if (gi > 0) {
                    const prev_mg = matched.items[gi - 1];
                    const tight_punct = isTightRightPunct(mg.cp) or isTightLeftPunct(prev_mg.cp) or
                        ((prev_mg.cp == '.' or prev_mg.cp == ':') and isAsciiDigit(mg.cp)) or
                        prev_mg.cp == '_' or mg.cp == '_';
                    const is_bullet = prev_mg.cp == 0x00B7 or prev_mg.cp == 0x2022 or mg.cp == 0x00B7 or mg.cp == 0x2022;
                    const narrow_punct = !is_bullet and ((prev_mg.width <= 3 and prev_mg.height <= 8) or (mg.width <= 3 and mg.height <= 8));
                    const narrow_digit_1 = isAsciiDigit(prev_mg.cp) and isAsciiDigit(mg.cp) and (prev_mg.cp == '1' or mg.cp == '1');
                    const narrow_pair = !is_bullet and ((prev_mg.width <= 3 and mg.width <= 3) or (med_w >= 7.0 and (prev_mg.width <= 3 or mg.width <= 3)));
                    const is_ellipsis = (prev_mg.cp == '.' and mg.cp == '.');
                    const eff_space_gap = if (is_ellipsis)
                        space_gap * 2.3
                    else if (tight_punct)
                        space_gap * 1.85
                    else if (narrow_punct or narrow_digit_1)
                        space_gap * 1.45
                    else if (narrow_pair)
                        space_gap + 1.0
                    else
                        space_gap;
                    if (mg.gap_before >= eff_space_gap) {
                        phrase_buf.append(gpa, ' ') catch return;
                    }
                }
                var utf8_buf: [4]u8 = undefined;
                const cp21: u21 = @intCast(mg.cp);
                const ulen = std.unicode.utf8Encode(cp21, &utf8_buf) catch 1;
                phrase_buf.appendSlice(gpa, utf8_buf[0..ulen]) catch return;
            }
            const mean_score = score_sum / @as(f32, @floatFromInt(char_cnt));
            var alnum_cnt: u32 = 0;
            for (matched.items) |mg| {
                if (isAsciiAlpha(mg.cp) or isAsciiDigit(mg.cp)) alnum_cnt += 1;
            }
            // Calibrated noise/garbage rejection gates (prevents hallucinating OCR_OBSERVED on noise/textures)
            if (mean_score < 0.52) continue;
            if (char_cnt == 1 and mean_score < 0.82) continue;
            if (char_cnt == 2 and (mean_score < 0.72 or (alnum_cnt == 0 and mean_score < 0.84))) continue;
            if (char_cnt >= 3 and alnum_cnt == 0 and mean_score < 0.80) continue;
            if (char_cnt >= 3 and alnum_cnt * 4 < char_cnt and mean_score < 0.74) continue;

            const owned = gpa.dupe(u8, phrase_buf.items) catch return;
            ctx.band_outs[bi].append(gpa, .{
                .x0 = px0,
                .y0 = py0,
                .x1 = px1,
                .y1 = py1,
                .mean_conf = mean_score * 100.0,
                .min_conf = min_score * 100.0,
                .text = owned,
            }) catch return;
        }
    }
}

/// Group components into horizontal text phrases, merge vertical diacritics, match shapes in parallel.
const StageTiming = struct {
    cc_ms: f64 = 0,
    band_ms: f64 = 0,
    match_ms: f64 = 0,
};

fn recognizeLines(
    gpa: std.mem.Allocator,
    io: std.Io,
    img: PgmImage,
    fg: []const u8,
    mask: []u8,
    atlas: Atlas,
    timing: *StageTiming,
) !std.ArrayList(OcrLine) {
    const t_cc0 = std.Io.Clock.now(.awake, io);
    var comps = try extractComponents(gpa, mask, img.width, img.height);
    defer comps.deinit(gpa);
    const t_cc1 = std.Io.Clock.now(.awake, io);
    timing.cc_ms = @as(f64, @floatFromInt(t_cc1.nanoseconds - t_cc0.nanoseconds)) / 1_000_000.0;

    var lines_out = try std.ArrayList(OcrLine).initCapacity(gpa, 128);
    errdefer {
        for (lines_out.items) |ln| gpa.free(ln.text);
        lines_out.deinit(gpa);
    }

    if (comps.items.len == 0) return lines_out;

    std.mem.sort(Comp, comps.items, {}, compLessByY);

    var bands = try std.ArrayList(Band).initCapacity(gpa, 128);
    defer {
        for (bands.items) |*b| b.indices.deinit(gpa);
        bands.deinit(gpa);
    }

    var small_indices = try std.ArrayList(usize).initCapacity(gpa, 256);
    defer small_indices.deinit(gpa);

    // Pass 1: Cluster normal-height glyphs (>= 5px) using baseline (y1) + descender-aware fit
    for (comps.items, 0..) |c, ci| {
        const ch = c.height();
        const cw = c.width();
        if (ch <= 5 and cw <= 14) {
            try small_indices.append(gpa, ci);
            continue;
        }
        const c_cy = c.cy();
        const c_y1: f32 = @floatFromInt(c.y1);

        var best_bi: ?usize = null;
        var best_err: f32 = 1e9;

        var bi: usize = bands.items.len;
        while (bi > 0) {
            bi -= 1;
            const b = &bands.items[bi];
            if (c.y0 > b.y1 + 12) break;

            const b_cy: f32 = b.core_sum_cy / @as(f32, @floatFromInt(b.core_cnt));
            const b_y1: f32 = b.core_sum_y1 / @as(f32, @floatFromInt(b.core_cnt));
            const b_h: f32 = std.math.clamp(b.core_sum_h / @as(f32, @floatFromInt(b.core_cnt)), 6.0, 22.0);

            const dy_base = @abs(c_y1 - b_y1);
            const dy_cy = @abs(c_cy - b_cy);
            const core_top = b_y1 - b_h * 1.25;
            const core_bot = b_y1 + b_h * 0.45;
            const ov_top = @max(@as(f32, @floatFromInt(c.y0)), core_top);
            const ov_bot = @min(@as(f32, @floatFromInt(c.y1)), core_bot);

            // Same line iff baselines align (within 3.5px), OR descender starts inside x-height and drops <= 0.5*b_h below baseline
            const same_baseline = dy_base <= @max(3.5, b_h * 0.30);
            const descender_fit = (c_y1 > b_y1 and c_y1 <= b_y1 + @max(5.5, b_h * 0.50) and
                @as(f32, @floatFromInt(c.y0)) >= @as(f32, @floatFromInt(b.y0)) and
                @as(f32, @floatFromInt(c.y0)) + 2.0 <= b_y1);
            const center_fit = dy_cy <= @max(4.0, b_h * 0.40);
            if (ov_bot > ov_top and (same_baseline or descender_fit or center_fit)) {
                const err = if (same_baseline) dy_base else @min(dy_base, dy_cy);
                if (err < best_err) {
                    best_err = err;
                    best_bi = bi;
                }
            }
        }

        if (best_bi) |idx_b| {
            const b = &bands.items[idx_b];
            if (c.x0 < b.x0) b.x0 = c.x0;
            if (c.x1 > b.x1) b.x1 = c.x1;
            if (c.y0 < b.y0 and (b.y1 - c.y0) <= 32) b.y0 = c.y0;
            if (c.y1 > b.y1 and (c.y1 - b.y0) <= 32) b.y1 = c.y1;
            if (ch <= 24) {
                b.core_sum_cy += c_cy;
                b.core_sum_y1 += c_y1;
                b.core_sum_h += @floatFromInt(ch);
                b.core_cnt += 1;
            }
            try b.indices.append(gpa, ci);
        } else {
            var idxs = try std.ArrayList(usize).initCapacity(gpa, 32);
            try idxs.append(gpa, ci);
            try bands.append(gpa, .{
                .x0 = c.x0,
                .x1 = c.x1,
                .y0 = c.y0,
                .y1 = c.y1,
                .core_sum_cy = c_cy,
                .core_sum_y1 = c_y1,
                .core_sum_h = @floatFromInt(@min(ch, 24)),
                .core_cnt = 1,
                .indices = idxs,
            });
        }
    }

    // Pass 1b: Re-unify any split ascender/descender bands sharing horizontal x-overlap and near-identical baseline (<= 4px)
    for (0..bands.items.len) |i| {
        if (bands.items[i].merged) continue;
        var j = i + 1;
        while (j < bands.items.len) : (j += 1) {
            if (bands.items[j].merged) continue;
            const bi = &bands.items[i];
            const bj = &bands.items[j];
            if (bj.y0 > bi.y1 + 6) break;
            const x_ov = @min(bi.x1, bj.x1) > @max(bi.x0, bj.x0);
            if (!x_ov) continue;
            const bi_y1 = bi.core_sum_y1 / @as(f32, @floatFromInt(bi.core_cnt));
            const bj_y1 = bj.core_sum_y1 / @as(f32, @floatFromInt(bj.core_cnt));
            if (@abs(bi_y1 - bj_y1) <= 4.0 and (@max(bi.y1, bj.y1) - @min(bi.y0, bj.y0)) <= 26) {
                try bi.indices.appendSlice(gpa, bj.indices.items);
                bi.x0 = @min(bi.x0, bj.x0);
                bi.x1 = @max(bi.x1, bj.x1);
                bi.y0 = @min(bi.y0, bj.y0);
                bi.y1 = @max(bi.y1, bj.y1);
                bi.core_sum_cy += bj.core_sum_cy;
                bi.core_sum_y1 += bj.core_sum_y1;
                bi.core_sum_h += bj.core_sum_h;
                bi.core_cnt += bj.core_cnt;
                bj.merged = true;
            }
        }
    }

    // Pass 2: Attach small components (accents, i-dots, hyphens, underscores, periods, commas) to nearest line band
    for (small_indices.items) |ci| {
        const c = comps.items[ci];
        const c_cy = c.cy();
        var best_bi: ?usize = null;
        var best_dist: f32 = 1e9;
        for (bands.items, 0..) |*b, bi| {
            if (b.merged) continue;
            if (c.y1 + 7 < b.y0 or c.y0 > b.y1 + 6) continue;
            const b_cy: f32 = b.core_sum_cy / @as(f32, @floatFromInt(b.core_cnt));
            const b_h: f32 = std.math.clamp(b.core_sum_h / @as(f32, @floatFromInt(b.core_cnt)), 6.0, 22.0);
            const dist = @abs(c_cy - b_cy);
            if (dist <= b_h * 1.05 + 5.0 and dist < best_dist) {
                best_dist = dist;
                best_bi = bi;
            }
        }
        if (best_bi) |idx_b| {
            try bands.items[idx_b].indices.append(gpa, ci);
        }
    }

    const t_m0 = std.Io.Clock.now(.awake, io);
    timing.band_ms = @as(f64, @floatFromInt(t_m0.nanoseconds - t_cc1.nanoseconds)) / 1_000_000.0;

    const band_outs = try gpa.alloc(std.ArrayList(OcrLine), bands.items.len);
    for (band_outs) |*bo| bo.* = .empty;
    defer {
        for (band_outs) |*bo| bo.deinit(gpa);
        gpa.free(band_outs);
    }

    // LPT scheduling: dispatch densest bands first to eliminate tail-latency stragglers
    const band_order = try gpa.alloc(usize, bands.items.len);
    defer gpa.free(band_order);
    var active_cnt: usize = 0;
    for (bands.items, 0..) |b, bi| {
        if (!b.merged and b.indices.items.len > 0) {
            band_order[active_cnt] = bi;
            active_cnt += 1;
        }
    }
    const SortBandCtx = struct {
        bands: []const Band,
        pub fn less(self: @This(), a: usize, b: usize) bool {
            return self.bands[a].indices.items.len > self.bands[b].indices.items.len;
        }
    };
    std.mem.sort(usize, band_order[0..active_cnt], SortBandCtx{ .bands = bands.items }, SortBandCtx.less);

    var ctx = BandWorkerCtx{
        .gpa = gpa,
        .img_w = img.width,
        .fg = fg,
        .mask = mask,
        .atlas = atlas,
        .comps = comps.items,
        .bands = bands.items,
        .band_order = band_order[0..active_cnt],
        .band_outs = band_outs,
        .next_task = std.atomic.Value(usize).init(0),
    };

    const cpu_cnt = std.Thread.getCpuCount() catch 4;
    const n_threads: usize = @max(1, @min(@min(@as(usize, 12), cpu_cnt), bands.items.len));
    var threads: [12]?std.Thread = .{null} ** 12;
    if (n_threads > 1) {
        for (0..n_threads - 1) |ti| {
            threads[ti] = std.Thread.spawn(.{}, bandWorkerRun, .{&ctx}) catch null;
        }
    }
    // Main thread participates as worker 0
    bandWorkerRun(&ctx);
    for (threads) |maybe_t| {
        if (maybe_t) |t| t.join();
    }

    // Flatten per-band results in deterministic original band order
    for (band_outs) |*bo| {
        try lines_out.appendSlice(gpa, bo.items);
    }

    const t_m1 = std.Io.Clock.now(.awake, io);
    timing.match_ms = @as(f64, @floatFromInt(t_m1.nanoseconds - t_m0.nanoseconds)) / 1_000_000.0;

    return lines_out;
}

fn writeJsonEscaped(w: *std.Io.Writer, s: []const u8) !void {
    try w.writeByte('"');
    for (s) |c| {
        switch (c) {
            '"' => try w.writeAll("\\\""),
            '\\' => try w.writeAll("\\\\"),
            '\n' => try w.writeAll("\\n"),
            '\r' => try w.writeAll("\\r"),
            '\t' => try w.writeAll("\\t"),
            else => {
                if (c < 0x20) {
                    try w.print("\\u{x:0>4}", .{c});
                } else {
                    try w.writeByte(c);
                }
            },
        }
    }
    try w.writeByte('"');
}

fn writeOcrJson(
    w: *std.Io.Writer,
    atlas_len: usize,
    lines: []const OcrLine,
    min_conf: f32,
    total_ms: f64,
    ocr_ms: f64,
) !void {
    const status = if (lines.len == 0) "OCR_EMPTY" else "OCR_OBSERVED";
    try w.print(
        "{{\"status\":\"{s}\",\"engine\":\"flowsight-ocr-zig/0.1\",\"templates\":{d},\"elapsed_ms\":{d:.2},\"ocr_ms\":{d:.2},\"items\":[",
        .{ status, atlas_len, total_ms, ocr_ms },
    );

    for (lines, 0..) |ln, idx| {
        if (idx > 0) try w.writeByte(',');
        const item_status = if (ln.mean_conf >= min_conf) "OCR_OBSERVED" else "OCR_UNCERTAIN";
        try w.writeAll("{\"text\":");
        try writeJsonEscaped(w, ln.text);
        try w.print(
            ",\"bbox\":[{d},{d},{d},{d}],\"confidence\":{d:.1},\"min_char_confidence\":{d:.1},\"status\":\"{s}\"}}",
            .{ ln.x0, ln.y0, ln.x1, ln.y1, ln.mean_conf, ln.min_conf, item_status },
        );
    }
    try w.writeAll("]}\n");
    try w.flush();
}

/// Persistent binary frame server (`--serve`):
/// Request header (16B LE): magic "FSQ1", width u32, height u32, channels u8 (1=gray,3=RGB), flags u8, min_conf_q8 u16.
/// Payload: width * height * channels raw u8 bytes.
/// Response: single NDJSON line flushed to stdout. Exits 0 cleanly on stdin EOF.
fn runServe(gpa: std.mem.Allocator, io: std.Io, atlas: Atlas, global_profile: bool) !void {
    var rbuf: [65536]u8 = undefined;
    var sr = std.Io.File.stdin().reader(io, &rbuf);
    const reader = &sr.interface;

    var out_buf: [65536]u8 = undefined;
    var fw = std.Io.File.stdout().writer(io, &out_buf);
    const w = &fw.interface;

    // Reusable buffers across frames (zero page-faults in steady state)
    var payload_buf: []u8 = &.{};
    defer if (payload_buf.len > 0) gpa.free(payload_buf);
    var gray_buf: []u8 = &.{};
    defer if (gray_buf.len > 0) gpa.free(gray_buf);
    var fg_buf: []u8 = &.{};
    defer if (fg_buf.len > 0) gpa.free(fg_buf);
    var mask_buf: []u8 = &.{};
    defer if (mask_buf.len > 0) gpa.free(mask_buf);

    while (true) {
        var hdr: [16]u8 = undefined;
        reader.readSliceAll(&hdr) catch |err| switch (err) {
            error.EndOfStream => return,
            else => return err,
        };

        const t0 = std.Io.Clock.now(.awake, io);

        if (!std.mem.eql(u8, hdr[0..4], "FSQ1")) {
            try w.writeAll("{\"status\":\"OCR_ERROR\",\"detail\":\"invalid FSQ1 frame magic\",\"items\":[]}\n");
            try w.flush();
            return error.InvalidFrameMagic;
        }
        const width = readU32Le(&hdr, 4);
        const height = readU32Le(&hdr, 8);
        const channels = hdr[12];
        const flags = hdr[13];
        const min_conf = @as(f32, @floatFromInt(readU16Le(&hdr, 14))) / 256.0;
        const do_profile = global_profile or ((flags & 1) != 0);

        if (width == 0 or height == 0 or (channels != 1 and channels != 3)) {
            try w.writeAll("{\"status\":\"OCR_ERROR\",\"detail\":\"invalid frame dimensions or channels\",\"items\":[]}\n");
            try w.flush();
            return error.InvalidFrameDims;
        }
        const n_pixels = @as(usize, width) * @as(usize, height);
        const payload_bytes = n_pixels * @as(usize, channels);
        if (payload_bytes > 64 * 1024 * 1024) {
            try w.writeAll("{\"status\":\"OCR_ERROR\",\"detail\":\"frame exceeds 64 MiB limit\",\"items\":[]}\n");
            try w.flush();
            return error.FrameTooLarge;
        }

        if (payload_buf.len < payload_bytes) {
            if (payload_buf.len > 0) gpa.free(payload_buf);
            payload_buf = try gpa.alloc(u8, payload_bytes);
        }
        try reader.readSliceAll(payload_buf[0..payload_bytes]);

        const t_rec0 = std.Io.Clock.now(.awake, io);

        const gray_slice: []const u8 = if (channels == 3) blk: {
            if (gray_buf.len < n_pixels) {
                if (gray_buf.len > 0) gpa.free(gray_buf);
                gray_buf = try gpa.alloc(u8, n_pixels);
            }
            rgbToGrayParallel(payload_buf[0..payload_bytes], gray_buf[0..n_pixels]);
            break :blk gray_buf[0..n_pixels];
        } else payload_buf[0..n_pixels];

        if (fg_buf.len < n_pixels) {
            if (fg_buf.len > 0) gpa.free(fg_buf);
            fg_buf = try gpa.alloc(u8, n_pixels);
        }
        if (mask_buf.len < n_pixels) {
            if (mask_buf.len > 0) gpa.free(mask_buf);
            mask_buf = try gpa.alloc(u8, n_pixels);
        }

        const img = PgmImage{ .width = width, .height = height, .pixels = gray_slice };
        computeForegroundInto(img, fg_buf[0..n_pixels], mask_buf[0..n_pixels]);
        const t_fg1 = std.Io.Clock.now(.awake, io);

        var stage_timing = StageTiming{};
        var lines = try recognizeLines(gpa, io, img, fg_buf[0..n_pixels], mask_buf[0..n_pixels], atlas, &stage_timing);
        defer {
            for (lines.items) |ln| gpa.free(ln.text);
            lines.deinit(gpa);
        }

        const t1 = std.Io.Clock.now(.awake, io);
        const total_ms = @as(f64, @floatFromInt(t1.nanoseconds - t0.nanoseconds)) / 1_000_000.0;
        const ocr_ms = @as(f64, @floatFromInt(t1.nanoseconds - t_rec0.nanoseconds)) / 1_000_000.0;

        if (do_profile) {
            const read_ms = @as(f64, @floatFromInt(t_rec0.nanoseconds - t0.nanoseconds)) / 1_000_000.0;
            const fg_ms = @as(f64, @floatFromInt(t_fg1.nanoseconds - t_rec0.nanoseconds)) / 1_000_000.0;
            var pbuf: [512]u8 = undefined;
            var pw = std.Io.File.stderr().writer(io, &pbuf);
            try pw.interface.print(
                "SERVE_PROFILE read={d:.2}ms rgb+fg={d:.2}ms cc={d:.2}ms band={d:.2}ms match={d:.2}ms total={d:.2}ms\n",
                .{ read_ms, fg_ms, stage_timing.cc_ms, stage_timing.band_ms, stage_timing.match_ms, total_ms },
            );
            try pw.interface.flush();
        }

        try writeOcrJson(w, atlas.meta.len, lines.items, min_conf, total_ms, ocr_ms);
    }
}

pub fn main(init: std.process.Init) !void {
    const gpa = init.gpa;
    const io = init.io;

    var atlas_path: ?[]const u8 = null;
    var pgm_path: ?[]const u8 = null;
    var min_conf: f32 = 60.0;
    var profile: bool = false;
    var serve_mode: bool = false;

    var arg_it = init.minimal.args.iterate();
    defer arg_it.deinit();
    _ = arg_it.next(); // program name

    while (arg_it.next()) |arg| {
        if (std.mem.eql(u8, arg, "--atlas")) {
            atlas_path = arg_it.next();
        } else if (std.mem.eql(u8, arg, "--pgm")) {
            pgm_path = arg_it.next();
        } else if (std.mem.eql(u8, arg, "--min-conf")) {
            if (arg_it.next()) |v| {
                min_conf = std.fmt.parseFloat(f32, v) catch 60.0;
            }
        } else if (std.mem.eql(u8, arg, "--profile")) {
            profile = true;
        } else if (std.mem.eql(u8, arg, "--serve")) {
            serve_mode = true;
        }
    }

    if (atlas_path == null or (!serve_mode and pgm_path == null)) {
        var ebuf: [512]u8 = undefined;
        var ew = std.Io.File.stderr().writer(io, &ebuf);
        try ew.interface.writeAll("usage: flowsight-ocr-zig --atlas <atlas.fsoa> (--pgm <image.pgm|-> | --serve) [--min-conf 60] [--profile]\n");
        try ew.interface.flush();
        std.process.exit(2);
    }

    const t0 = std.Io.Clock.now(.awake, io);

    const atlas_raw = try std.Io.Dir.cwd().readFileAlloc(io, atlas_path.?, gpa, .limited(32 * 1024 * 1024));
    defer gpa.free(atlas_raw);
    const atlas = try loadAtlas(gpa, atlas_raw);
    defer atlas.deinit(gpa);
    const t_atlas1 = std.Io.Clock.now(.awake, io);

    if (serve_mode) {
        try runServe(gpa, io, atlas, profile);
        return;
    }

    const pgm_raw = if (std.mem.eql(u8, pgm_path.?, "-")) blk: {
        var rbuf: [65536]u8 = undefined;
        var sr = std.Io.File.stdin().reader(io, &rbuf);
        break :blk try sr.interface.allocRemaining(gpa, .limited(64 * 1024 * 1024));
    } else try std.Io.Dir.cwd().readFileAlloc(io, pgm_path.?, gpa, .limited(64 * 1024 * 1024));
    defer gpa.free(pgm_raw);
    const img = try parsePgm(pgm_raw);

    const t_rec0 = std.Io.Clock.now(.awake, io);

    const fg_res = try computeForeground(gpa, img);
    defer gpa.free(fg_res.fg);
    defer gpa.free(fg_res.mask);
    const t_fg1 = std.Io.Clock.now(.awake, io);

    var stage_timing = StageTiming{};
    var lines = try recognizeLines(gpa, io, img, fg_res.fg, fg_res.mask, atlas, &stage_timing);
    defer {
        for (lines.items) |ln| gpa.free(ln.text);
        lines.deinit(gpa);
    }

    const t1 = std.Io.Clock.now(.awake, io);
    const total_ms = @as(f64, @floatFromInt(t1.nanoseconds - t0.nanoseconds)) / 1_000_000.0;
    const ocr_ms = @as(f64, @floatFromInt(t1.nanoseconds - t_rec0.nanoseconds)) / 1_000_000.0;

    if (profile) {
        const atlas_ms = @as(f64, @floatFromInt(t_atlas1.nanoseconds - t0.nanoseconds)) / 1_000_000.0;
        const pgm_ms = @as(f64, @floatFromInt(t_rec0.nanoseconds - t_atlas1.nanoseconds)) / 1_000_000.0;
        const fg_ms = @as(f64, @floatFromInt(t_fg1.nanoseconds - t_rec0.nanoseconds)) / 1_000_000.0;
        var pbuf: [512]u8 = undefined;
        var pw = std.Io.File.stderr().writer(io, &pbuf);
        try pw.interface.print(
            "PROFILE atlas={d:.2}ms pgm={d:.2}ms fg={d:.2}ms cc={d:.2}ms band={d:.2}ms match={d:.2}ms total={d:.2}ms\n",
            .{ atlas_ms, pgm_ms, fg_ms, stage_timing.cc_ms, stage_timing.band_ms, stage_timing.match_ms, total_ms },
        );
        try pw.interface.flush();
    }

    var out_buf: [65536]u8 = undefined;
    var fw = std.Io.File.stdout().writer(io, &out_buf);
    try writeOcrJson(&fw.interface, atlas.meta.len, lines.items, min_conf, total_ms, ocr_ms);
}

test "parsePgm parses valid P5 header and pixels" {
    const raw = "P5\n2 2\n255\n\x00\xff\x80\x40";
    const img = try parsePgm(raw);
    try std.testing.expectEqual(@as(u32, 2), img.width);
    try std.testing.expectEqual(@as(u32, 2), img.height);
    try std.testing.expectEqual(@as(u8, 0xff), img.pixels[1]);
}

test "rgbToGrayParallel matches BT.601 luma" {
    const rgb = [_]u8{
        255, 0,   0,
        0,   255, 0,
        0,   0,   255,
        255, 255, 255,
        0,   0,   0,
    };
    var gray: [5]u8 = undefined;
    rgbToGrayParallel(&rgb, &gray);
    try std.testing.expectEqual(@as(u8, 76), gray[0]);
    try std.testing.expectEqual(@as(u8, 150), gray[1]);
    try std.testing.expectEqual(@as(u8, 29), gray[2]);
    try std.testing.expectEqual(@as(u8, 255), gray[3]);
    try std.testing.expectEqual(@as(u8, 0), gray[4]);
}

test "countHolesGlyph detects 0 holes in bar and 1 hole in ring" {
    // 5x5 solid ring with 2x2 hollow center (1 hole)
    const ring = [_]u8{
        1, 1, 1, 1, 1,
        1, 0, 0, 1, 1,
        1, 0, 0, 1, 1,
        1, 1, 1, 1, 1,
        0, 0, 0, 0, 0,
    };
    try std.testing.expectEqual(@as(u8, 1), countHolesGlyph(&ring, 5, 0, 0, 5, 4));
}

test "disambiguatePhrase fixes 2O26 -> 2026 and sal1da -> salida" {
    var num_tok = [_]MatchedGlyph{
        .{ .cp = '2', .score = 0.9, .second_cp = 'Z', .second_score = 0.6, .gap_before = 0, .width = 7, .height = 11 },
        .{ .cp = 'O', .score = 0.88, .second_cp = '0', .second_score = 0.85, .gap_before = 1, .width = 7, .height = 11 },
        .{ .cp = '2', .score = 0.9, .second_cp = 'Z', .second_score = 0.6, .gap_before = 1, .width = 7, .height = 11 },
        .{ .cp = '6', .score = 0.9, .second_cp = 'G', .second_score = 0.6, .gap_before = 1, .width = 7, .height = 11 },
    };
    disambiguatePhrase(&num_tok, 4.0, 10.0, 21.0);
    try std.testing.expectEqual(@as(u32, '0'), num_tok[1].cp);

    var word_tok = [_]MatchedGlyph{
        .{ .cp = 's', .score = 0.9, .second_cp = 'S', .second_score = 0.7, .gap_before = 0, .width = 6, .height = 8 },
        .{ .cp = 'a', .score = 0.9, .second_cp = 'o', .second_score = 0.7, .gap_before = 1, .width = 6, .height = 8 },
        .{ .cp = 'l', .score = 0.9, .second_cp = '1', .second_score = 0.8, .gap_before = 1, .width = 3, .height = 11 },
        .{ .cp = '1', .score = 0.85, .second_cp = 'i', .second_score = 0.82, .gap_before = 1, .width = 3, .height = 11 },
        .{ .cp = 'd', .score = 0.9, .second_cp = 'b', .second_score = 0.6, .gap_before = 1, .width = 6, .height = 11 },
        .{ .cp = 'a', .score = 0.9, .second_cp = 'o', .second_score = 0.7, .gap_before = 1, .width = 6, .height = 8 },
    };
    disambiguatePhrase(&word_tok, 4.0, 10.0, 21.0);
    try std.testing.expectEqual(@as(u32, 'i'), word_tok[3].cp);

    var case_tok = [_]MatchedGlyph{
        .{ .cp = 'B', .score = 0.9, .second_cp = '8', .second_score = 0.6, .gap_before = 0, .width = 7, .height = 11 },
        .{ .cp = 'u', .score = 0.9, .second_cp = 'U', .second_score = 0.7, .gap_before = 1, .width = 6, .height = 8 },
        .{ .cp = 'S', .score = 0.88, .second_cp = 's', .second_score = 0.86, .gap_before = 1, .width = 6, .height = 8 },
        .{ .cp = 'C', .score = 0.88, .second_cp = 'c', .second_score = 0.86, .gap_before = 1, .width = 6, .height = 8 },
        .{ .cp = 'a', .score = 0.9, .second_cp = 'o', .second_score = 0.7, .gap_before = 1, .width = 6, .height = 8 },
        .{ .cp = 'r', .score = 0.9, .second_cp = 't', .second_score = 0.7, .gap_before = 1, .width = 4, .height = 8 },
    };
    disambiguatePhrase(&case_tok, 4.0, 10.0, 21.0);
    try std.testing.expectEqual(@as(u32, 's'), case_tok[2].cp);
    try std.testing.expectEqual(@as(u32, 'c'), case_tok[3].cp);

    // FIowS¡8ht -> FlowSight (I->l after cap before lower, ¡->i inside word, 8->g inside word)
    var fs_tok = [_]MatchedGlyph{
        .{ .cp = 'F', .score = 0.9, .second_cp = 'E', .second_score = 0.6, .gap_before = 0, .width = 6, .height = 11, .y0 = 10, .y1 = 21 },
        .{ .cp = 'I', .score = 0.9, .second_cp = 'l', .second_score = 0.89, .gap_before = 1, .width = 2, .height = 11, .y0 = 10, .y1 = 21 },
        .{ .cp = 'o', .score = 0.9, .second_cp = 'O', .second_score = 0.7, .gap_before = 1, .width = 6, .height = 8, .y0 = 13, .y1 = 21 },
        .{ .cp = 'w', .score = 0.9, .second_cp = 'W', .second_score = 0.7, .gap_before = 1, .width = 8, .height = 8, .y0 = 13, .y1 = 21 },
        .{ .cp = 'S', .score = 0.9, .second_cp = 's', .second_score = 0.7, .gap_before = 1, .width = 6, .height = 11, .y0 = 10, .y1 = 21 },
        .{ .cp = 0x00A1, .score = 0.85, .second_cp = 'i', .second_score = 0.84, .gap_before = 1, .width = 2, .height = 11, .y0 = 10, .y1 = 21 },
        .{ .cp = '8', .score = 0.84, .second_cp = 'g', .second_score = 0.82, .gap_before = 1, .width = 6, .height = 11, .y0 = 13, .y1 = 24 },
        .{ .cp = 'h', .score = 0.9, .second_cp = 'n', .second_score = 0.6, .gap_before = 1, .width = 6, .height = 11, .y0 = 10, .y1 = 21 },
        .{ .cp = 't', .score = 0.9, .second_cp = 'f', .second_score = 0.6, .gap_before = 1, .width = 4, .height = 10, .y0 = 11, .y1 = 21 },
    };
    disambiguatePhrase(&fs_tok, 4.0, 10.0, 21.0);
    try std.testing.expectEqual(@as(u32, 'l'), fs_tok[1].cp);
    try std.testing.expectEqual(@as(u32, 'i'), fs_tok[5].cp);
    try std.testing.expectEqual(@as(u32, 'g'), fs_tok[6].cp);
}

test "resolveSameShapeCase keeps capitals when ascenders are taller than caps (DejaVu 13px)" {
    // Heights measured on DejaVuSans 13px "Cosas Sobre Otros": caps 9, x-height 7, 'b' ascender 11.
    var tok = [_]MatchedGlyph{
        .{ .cp = 'c', .score = 0.9, .second_cp = 'C', .second_score = 0.89, .gap_before = 0, .width = 7, .height = 9 },
        .{ .cp = 'o', .score = 0.9, .second_cp = 'O', .second_score = 0.89, .gap_before = 1, .width = 6, .height = 7 },
        .{ .cp = 'S', .score = 0.9, .second_cp = 's', .second_score = 0.89, .gap_before = 1, .width = 5, .height = 7 },
        .{ .cp = 'a', .score = 0.9, .second_cp = 'o', .second_score = 0.6, .gap_before = 1, .width = 6, .height = 7 },
        .{ .cp = 's', .score = 0.9, .second_cp = 'S', .second_score = 0.89, .gap_before = 1, .width = 5, .height = 7 },
        .{ .cp = 's', .score = 0.9, .second_cp = 'S', .second_score = 0.89, .gap_before = 6, .width = 6, .height = 9 },
        .{ .cp = 'o', .score = 0.9, .second_cp = 'O', .second_score = 0.89, .gap_before = 1, .width = 6, .height = 7 },
        .{ .cp = 'b', .score = 0.9, .second_cp = 'h', .second_score = 0.6, .gap_before = 1, .width = 6, .height = 11 },
        .{ .cp = 'r', .score = 0.9, .second_cp = 't', .second_score = 0.6, .gap_before = 1, .width = 4, .height = 7 },
        .{ .cp = 'e', .score = 0.9, .second_cp = 'c', .second_score = 0.6, .gap_before = 1, .width = 6, .height = 7 },
    };
    disambiguatePhrase(&tok, 4.0, 10.0, 21.0);
    try std.testing.expectEqual(@as(u32, 'C'), tok[0].cp);
    try std.testing.expectEqual(@as(u32, 'o'), tok[1].cp);
    try std.testing.expectEqual(@as(u32, 's'), tok[2].cp);
    try std.testing.expectEqual(@as(u32, 'S'), tok[5].cp);
}
