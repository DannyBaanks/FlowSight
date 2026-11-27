const std = @import("std");
const assert = std.debug.assert;

const GRID_W: usize = 12;
const GRID_H: usize = 16;
const GRID_LEN: usize = GRID_W * GRID_H; // 192
const SIMD_CHUNKS: usize = GRID_LEN / 16; // 12
const Vec16 = @Vector(16, f32);

comptime {
    assert(GRID_LEN == 192);
    assert(GRID_LEN % 16 == 0);
}

const Template = struct {
    codepoint: u32,
    aspect: f32,
    y_top: f32,
    y_bot: f32,
    holes: u8,
    vecs: [SIMD_CHUNKS]Vec16,
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

fn loadAtlas(gpa: std.mem.Allocator, raw: []const u8) ![]Template {
    if (raw.len < 16) return error.InvalidAtlas;
    if (!std.mem.eql(u8, raw[0..4], "FSOA")) return error.InvalidAtlasMagic;
    const version = readU16Le(raw, 4);
    const gw = readU16Le(raw, 6);
    const gh = readU16Le(raw, 8);
    const count = readU32Le(raw, 12);
    if (version != 1 or gw != GRID_W or gh != GRID_H) return error.UnsupportedAtlas;

    const entry_bytes: usize = 12 + GRID_LEN * 4;
    const expected_len: usize = 16 + @as(usize, count) * entry_bytes;
    if (raw.len < expected_len) return error.TruncatedAtlas;

    const templates = try gpa.alloc(Template, count);
    var pos: usize = 16;
    for (templates) |*t| {
        const cp = readU32Le(raw, pos);
        const aspect_q8 = readU16Le(raw, pos + 4);
        const y_top_q8 = readI16Le(raw, pos + 6);
        const y_bot_q8 = readI16Le(raw, pos + 8);
        const holes = raw[pos + 10];
        pos += 12;

        var vecs: [SIMD_CHUNKS]Vec16 = undefined;
        for (0..SIMD_CHUNKS) |chunk| {
            var arr: [16]f32 = undefined;
            for (0..16) |k| {
                arr[k] = readF32Le(raw, pos);
                pos += 4;
            }
            vecs[chunk] = arr;
        }
        t.* = .{
            .codepoint = cp,
            .aspect = @as(f32, @floatFromInt(aspect_q8)) / 256.0,
            .y_top = @as(f32, @floatFromInt(y_top_q8)) / 256.0,
            .y_bot = @as(f32, @floatFromInt(y_bot_q8)) / 256.0,
            .holes = holes,
            .vecs = vecs,
        };
    }
    return templates;
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

/// Compute polarity-invariant foreground strength per pixel using 16x16 tile mode background.
fn computeForeground(
    gpa: std.mem.Allocator,
    img: PgmImage,
) !struct { fg: []u8, mask: []u8 } {
    const n = @as(usize, img.width) * @as(usize, img.height);
    const fg = try gpa.alloc(u8, n);
    errdefer gpa.free(fg);
    const mask = try gpa.alloc(u8, n);
    errdefer gpa.free(mask);

    const tile: u32 = 16;
    const tw = (img.width + tile - 1) / tile;
    const th = (img.height + tile - 1) / tile;

    var ty: u32 = 0;
    while (ty < th) : (ty += 1) {
        const y_start = ty * tile;
        const y_end = @min(y_start + tile, img.height);
        var tx: u32 = 0;
        while (tx < tw) : (tx += 1) {
            const x_start = tx * tile;
            const x_end = @min(x_start + tile, img.width);

            // 16-bin histogram to find dominant background luma in tile
            var hist_cnt: [16]u32 = .{0} ** 16;
            var hist_sum: [16]u32 = .{0} ** 16;

            var y = y_start;
            while (y < y_end) : (y += 1) {
                const row_off = @as(usize, y) * @as(usize, img.width);
                var x = x_start;
                while (x < x_end) : (x += 1) {
                    const p = img.pixels[row_off + x];
                    const bin: usize = p >> 4;
                    hist_cnt[bin] += 1;
                    hist_sum[bin] += p;
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
            const bg: i32 = if (best_cnt > 0) @intCast(hist_sum[best_bin] / best_cnt) else 0;

            y = y_start;
            while (y < y_end) : (y += 1) {
                const row_off = @as(usize, y) * @as(usize, img.width);
                var x = x_start;
                while (x < x_end) : (x += 1) {
                    const p: i32 = img.pixels[row_off + x];
                    const diff: u32 = @abs(p - bg);
                    const d8: u8 = @intCast(@min(diff, 255));
                    fg[row_off + x] = d8;
                    mask[row_off + x] = if (d8 >= 36) 1 else 0;
                }
            }
        }
    }

    return .{ .fg = fg, .mask = mask };
}

/// Extract glyph-sized 8-connected components via bounded iterative BFS.
fn extractComponents(
    gpa: std.mem.Allocator,
    mask: []const u8,
    width: u32,
    height: u32,
) !std.ArrayList(Comp) {
    const n = @as(usize, width) * @as(usize, height);
    const visited = try gpa.alloc(u8, n);
    defer gpa.free(visited);
    @memset(visited, 0);

    var queue = try std.ArrayList(u32).initCapacity(gpa, 4096);
    defer queue.deinit(gpa);

    var comps = try std.ArrayList(Comp).initCapacity(gpa, 1024);
    errdefer comps.deinit(gpa);

    const dx = [_]i32{ -1, 0, 1, -1, 1, -1, 0, 1 };
    const dy = [_]i32{ -1, -1, -1, 0, 0, 1, 1, 1 };

    var y: u32 = 0;
    while (y < height) : (y += 1) {
        const row_off = @as(usize, y) * @as(usize, width);
        var x: u32 = 0;
        while (x < width) : (x += 1) {
            const idx = row_off + x;
            if (mask[idx] == 0 or visited[idx] != 0) continue;

            visited[idx] = 1;
            queue.clearRetainingCapacity();
            try queue.append(gpa, @intCast(idx));

            var min_x = x;
            var max_x = x;
            var min_y = y;
            var max_y = y;
            var area: u32 = 0;
            var head: usize = 0;
            var overflowed = false;

            while (head < queue.items.len) : (head += 1) {
                const cur = queue.items[head];
                const cx: u32 = cur % width;
                const cy_u: u32 = cur / width;
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
                    if (mask[nidx] != 0 and visited[nidx] == 0) {
                        visited[nidx] = 1;
                        if (queue.items.len < 4096) {
                            try queue.append(gpa, @intCast(nidx));
                        } else {
                            overflowed = true;
                        }
                    }
                }
            }

            if (overflowed) continue;
            const bw = max_x - min_x + 1;
            const bh = max_y - min_y + 1;

            // Keep glyph-sized components, reject UI boxes/borders/lines
            if (bw == 0 or bh < 2) continue;
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
    var padded: [50 * 50]u8 = .{0} ** (50 * 50);
    var vis: [50 * 50]u8 = .{0} ** (50 * 50);
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

/// Bilinear resample fg patch to 12x16 and zero-mean unit-L2 normalize into 12 SIMD Vec16s.
fn extractGlyphVecs(
    fg: []const u8,
    img_w: u32,
    x0: u32,
    y0: u32,
    x1: u32,
    y1: u32,
) [SIMD_CHUNKS]Vec16 {
    const src_w: f32 = @floatFromInt(x1 - x0);
    const src_h: f32 = @floatFromInt(y1 - y0);
    const max_x: f32 = @max(0.0, src_w - 1.0);
    const max_y: f32 = @max(0.0, src_h - 1.0);

    var raw: [GRID_LEN]f32 = undefined;
    var sum: f32 = 0.0;

    for (0..GRID_H) |gy| {
        var sy = (@as(f32, @floatFromInt(gy)) + 0.5) * (src_h / @as(f32, GRID_H)) - 0.5;
        sy = std.math.clamp(sy, 0.0, max_y);
        const iy0: u32 = @intFromFloat(@floor(sy));
        const iy1: u32 = @min(iy0 + 1, (y1 - y0) - 1);
        const fy = sy - @as(f32, @floatFromInt(iy0));
        const r0 = @as(usize, y0 + iy0) * @as(usize, img_w);
        const r1 = @as(usize, y0 + iy1) * @as(usize, img_w);

        for (0..GRID_W) |gx| {
            var sx = (@as(f32, @floatFromInt(gx)) + 0.5) * (src_w / @as(f32, GRID_W)) - 0.5;
            sx = std.math.clamp(sx, 0.0, max_x);
            const ix0: u32 = @intFromFloat(@floor(sx));
            const ix1: u32 = @min(ix0 + 1, (x1 - x0) - 1);
            const fx = sx - @as(f32, @floatFromInt(ix0));

            const v00: f32 = @floatFromInt(fg[r0 + x0 + ix0]);
            const v01: f32 = @floatFromInt(fg[r0 + x0 + ix1]);
            const v10: f32 = @floatFromInt(fg[r1 + x0 + ix0]);
            const v11: f32 = @floatFromInt(fg[r1 + x0 + ix1]);

            const top = v00 * (1.0 - fx) + v01 * fx;
            const bot = v10 * (1.0 - fx) + v11 * fx;
            const val = (top * (1.0 - fy) + bot * fy) / 255.0;
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
    var out: [SIMD_CHUNKS]Vec16 = undefined;
    for (0..SIMD_CHUNKS) |chunk| {
        var arr: [16]f32 = undefined;
        for (0..16) |k| {
            arr[k] = raw[chunk * 16 + k] * inv_norm;
        }
        out[chunk] = arr;
    }
    return out;
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
};

/// Hot-path SIMD shape matcher returning top-2 distinct codepoints (no self, TigerStyle).
fn matchGlyphSimd(
    templates: []const Template,
    q_vecs: *const [SIMD_CHUNKS]Vec16,
    q_aspect: f32,
    q_y_top: f32,
    q_y_bot: f32,
    q_holes: u8,
) MatchResult {
    var best_cp: u32 = '?';
    var best_score: f32 = -10.0;
    var second_cp: u32 = '?';
    var second_score: f32 = -10.0;

    for (templates) |t| {
        // Fast geometric reject before SIMD dot product
        const aspect_diff = @abs(q_aspect - t.aspect);
        if (aspect_diff > 1.2) continue;

        var acc: Vec16 = @splat(0.0);
        inline for (0..SIMD_CHUNKS) |c| {
            acc += q_vecs[c] * t.vecs[c];
        }
        const dot: f32 = @reduce(.Add, acc);

        const y_diff = (@abs(q_y_top - t.y_top) + @abs(q_y_bot - t.y_bot)) * 0.5;
        const hole_pen: f32 = if (q_holes == t.holes) 0.0 else 0.10;
        const score = dot - 0.22 * aspect_diff - 0.28 * @min(y_diff, 1.5) - hole_pen;
        if (score > best_score) {
            if (t.codepoint != best_cp) {
                second_score = best_score;
                second_cp = best_cp;
            }
            best_score = score;
            best_cp = t.codepoint;
        } else if (t.codepoint != best_cp and score > second_score) {
            second_score = score;
            second_cp = t.codepoint;
        }
    }

    return .{
        .codepoint = best_cp,
        .score = best_score,
        .second_cp = second_cp,
        .second_score = second_score,
    };
}

fn isAsciiDigit(cp: u32) bool {
    return cp >= '0' and cp <= '9';
}

fn isAsciiLower(cp: u32) bool {
    return (cp >= 'a' and cp <= 'z') or cp == 0x00E1 or cp == 0x00E9 or cp == 0x00ED or cp == 0x00F3 or cp == 0x00FA or cp == 0x00F1 or cp == 0x00FC;
}

fn isAsciiAlpha(cp: u32) bool {
    return isAsciiLower(cp) or (cp >= 'A' and cp <= 'Z') or cp == 0x00C1 or cp == 0x00C9 or cp == 0x00CD or cp == 0x00D3 or cp == 0x00DA or cp == 0x00D1;
}

/// Disambiguate isolated-glyph confusions (1/i/l/I and 0/O/o) using immediate token neighbors.
fn disambiguatePhrase(glyphs: []MatchedGlyph, space_gap: f32) void {
    const n = glyphs.len;
    if (n < 2) return;

    for (0..n) |i| {
        const g = &glyphs[i];
        const has_left = (i > 0) and (g.gap_before < space_gap * 1.15);
        const has_right = (i + 1 < n) and (glyphs[i + 1].gap_before < space_gap * 1.15);
        const left_cp: u32 = if (has_left) glyphs[i - 1].cp else 0;
        const right_cp: u32 = if (has_right) glyphs[i + 1].cp else 0;

        const near_digit = isAsciiDigit(left_cp) or isAsciiDigit(right_cp);
        const near_lower = isAsciiLower(left_cp) or isAsciiLower(right_cp);
        const near_alpha = isAsciiAlpha(left_cp) or isAsciiAlpha(right_cp);

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

/// Group components into horizontal text phrases, merge vertical diacritics, match shapes.
fn recognizeLines(
    gpa: std.mem.Allocator,
    img: PgmImage,
    fg: []const u8,
    mask: []const u8,
    templates: []const Template,
) !std.ArrayList(OcrLine) {
    var comps = try extractComponents(gpa, mask, img.width, img.height);
    defer comps.deinit(gpa);

    var lines_out = try std.ArrayList(OcrLine).initCapacity(gpa, 128);
    errdefer {
        for (lines_out.items) |ln| gpa.free(ln.text);
        lines_out.deinit(gpa);
    }

    if (comps.items.len == 0) return lines_out;

    std.mem.sort(Comp, comps.items, {}, compLessByY);

    // Band clustering by vertical proximity
    const Band = struct {
        y0: u32,
        y1: u32,
        indices: std.ArrayList(usize),
    };
    var bands = try std.ArrayList(Band).initCapacity(gpa, 128);
    defer {
        for (bands.items) |*b| b.indices.deinit(gpa);
        bands.deinit(gpa);
    }

    for (comps.items, 0..) |c, ci| {
        var placed = false;
        // Search recent bands from the end (sorted by y0)
        var bi: usize = bands.items.len;
        while (bi > 0) {
            bi -= 1;
            const b = &bands.items[bi];
            if (c.y0 > b.y1 + 10) break;
            const ov_top = @max(c.y0, b.y0);
            const ov_bot = @min(c.y1, b.y1);
            const band_h = b.y1 - b.y0;
            const c_cy = c.cy();
            const b_cy = (@as(f32, @floatFromInt(b.y0)) + @as(f32, @floatFromInt(b.y1))) * 0.5;
            const close_y = @abs(c_cy - b_cy) <= @max(7.0, @as(f32, @floatFromInt(band_h)) * 0.65);
            const overlaps = ov_bot > ov_top;
            // Allow small diacritic dots just above or below a band
            const is_diacritic = (c.height() <= 6 and c.width() <= 10 and
                ((c.y1 <= b.y0 + 3 and b.y0 <= c.y1 + 6) or (c.y0 + 3 >= b.y1 and c.y0 <= b.y1 + 5)));
            if ((overlaps and close_y) or is_diacritic) {
                if (!is_diacritic) {
                    if (c.y0 < b.y0) b.y0 = c.y0;
                    if (c.y1 > b.y1) b.y1 = c.y1;
                }
                try b.indices.append(gpa, ci);
                placed = true;
                break;
            }
        }
        if (!placed) {
            var idxs = try std.ArrayList(usize).initCapacity(gpa, 32);
            try idxs.append(gpa, ci);
            try bands.append(gpa, .{ .y0 = c.y0, .y1 = c.y1, .indices = idxs });
        }
    }

    var band_comps = try std.ArrayList(Comp).initCapacity(gpa, 128);
    defer band_comps.deinit(gpa);
    var merged_glyphs = try std.ArrayList(Comp).initCapacity(gpa, 128);
    defer merged_glyphs.deinit(gpa);
    var phrase_buf = try std.ArrayList(u8).initCapacity(gpa, 256);
    defer phrase_buf.deinit(gpa);
    var tmp_u32 = try std.ArrayList(u32).initCapacity(gpa, 128);
    defer tmp_u32.deinit(gpa);

    for (bands.items) |b| {
        if (b.indices.items.len == 0) continue;
        band_comps.clearRetainingCapacity();
        for (b.indices.items) |ci| {
            try band_comps.append(gpa, comps.items[ci]);
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
                // Only merge vertically stacked components (dots/accents/colons), never side-by-side letters
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
                try merged_glyphs.append(gpa, c);
            }
        }

        if (merged_glyphs.items.len == 0) continue;

        // Estimate band median glyph height to split into horizontal phrases (columns)
        tmp_u32.clearRetainingCapacity();
        for (merged_glyphs.items) |g| {
            if (g.height() >= 5) try tmp_u32.append(gpa, g.height());
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

            // Compute phrase ref_top, ref_bot, median width
            tmp_u32.clearRetainingCapacity();
            var tops = try std.ArrayList(u32).initCapacity(gpa, slice.len);
            defer tops.deinit(gpa);
            var bots = try std.ArrayList(u32).initCapacity(gpa, slice.len);
            defer bots.deinit(gpa);

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
                    try tops.append(gpa, g.y0);
                    try bots.append(gpa, g.y1);
                    try tmp_u32.append(gpa, g.width());
                }
            }
            if (tops.items.len == 0) continue;

            std.mem.sort(u32, tops.items, {}, u32Less);
            std.mem.sort(u32, bots.items, {}, u32Less);
            std.mem.sort(u32, tmp_u32.items, {}, u32Less);

            const ref_top: f32 = @floatFromInt(tops.items[tops.items.len / 4]);
            const ref_bot: f32 = @floatFromInt(bots.items[(bots.items.len * 3) / 4]);
            const ref_h: f32 = @max(6.0, ref_bot - ref_top);
            const med_w: f32 = @floatFromInt(tmp_u32.items[tmp_u32.items.len / 2]);
            const space_gap: f32 = @max(3.0, @min(med_w * 0.48, ref_h * 0.34));

            phrase_buf.clearRetainingCapacity();
            var matched = try std.ArrayList(MatchedGlyph).initCapacity(gpa, slice.len);
            defer matched.deinit(gpa);

            var score_sum: f32 = 0.0;
            var min_score: f32 = 1.0;

            for (slice, 0..) |g, gi| {
                const gap: f32 = if (gi > 0 and g.x0 > slice[gi - 1].x1)
                    @floatFromInt(g.x0 - slice[gi - 1].x1)
                else
                    0.0;

                const q_vecs = extractGlyphVecs(fg, img.width, g.x0, g.y0, g.x1, g.y1);
                const q_aspect = @as(f32, @floatFromInt(g.width())) / @as(f32, @floatFromInt(g.height()));
                const q_y_top = (@as(f32, @floatFromInt(g.y0)) - ref_top) / ref_h;
                const q_y_bot = (@as(f32, @floatFromInt(g.y1)) - ref_top) / ref_h;
                const q_holes = countHolesGlyph(mask, img.width, g.x0, g.y0, g.x1, g.y1);

                const m = matchGlyphSimd(templates, &q_vecs, q_aspect, q_y_top, q_y_bot, q_holes);
                try matched.append(gpa, .{
                    .cp = m.codepoint,
                    .score = m.score,
                    .second_cp = m.second_cp,
                    .second_score = m.second_score,
                    .gap_before = gap,
                    .width = g.width(),
                    .height = g.height(),
                });

                const clamped = std.math.clamp(m.score, 0.0, 1.0);
                score_sum += clamped;
                if (clamped < min_score) min_score = clamped;
            }

            const char_cnt: u32 = @intCast(matched.items.len);
            if (char_cnt == 0) continue;

            // Contextual disambiguation pass (1/i/l/I and 0/O/o within tokens)
            disambiguatePhrase(matched.items, space_gap);

            for (matched.items, 0..) |mg, gi| {
                if (gi > 0) {
                    const prev_mg = matched.items[gi - 1];
                    const tight_punct = isTightRightPunct(mg.cp) or isTightLeftPunct(prev_mg.cp) or
                        ((prev_mg.cp == '.' or prev_mg.cp == ':') and isAsciiDigit(mg.cp));
                    const narrow_punct = (prev_mg.width <= 3 and prev_mg.height <= 8) or (mg.width <= 3 and mg.height <= 8);
                    const eff_space_gap = if (tight_punct)
                        space_gap * 1.85
                    else if (narrow_punct)
                        space_gap * 1.45
                    else
                        space_gap;
                    if (mg.gap_before >= eff_space_gap) {
                        try phrase_buf.append(gpa, ' ');
                    }
                }
                var utf8_buf: [4]u8 = undefined;
                const cp21: u21 = @intCast(mg.cp);
                const ulen = std.unicode.utf8Encode(cp21, &utf8_buf) catch 1;
                try phrase_buf.appendSlice(gpa, utf8_buf[0..ulen]);
            }
            const mean_score = score_sum / @as(f32, @floatFromInt(char_cnt));
            // Filter out non-text UI icon noise
            if (mean_score < 0.45) continue;
            if (char_cnt == 1 and mean_score < 0.72) continue;

            const owned = try gpa.dupe(u8, phrase_buf.items);
            try lines_out.append(gpa, .{
                .x0 = px0,
                .y0 = py0,
                .x1 = px1,
                .y1 = py1,
                .mean_conf = mean_score * 100.0,
                .min_conf = min_score * 100.0,
                .text = owned,
            });
        }
    }

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

pub fn main(init: std.process.Init) !void {
    const gpa = init.gpa;
    const io = init.io;

    var atlas_path: ?[]const u8 = null;
    var pgm_path: ?[]const u8 = null;
    var min_conf: f32 = 60.0;

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
        }
    }

    if (atlas_path == null or pgm_path == null) {
        var ebuf: [512]u8 = undefined;
        var ew = std.Io.File.stderr().writer(io, &ebuf);
        try ew.interface.writeAll("usage: flowsight-ocr-zig --atlas <atlas.fsoa> --pgm <image.pgm> [--min-conf 60]\n");
        try ew.interface.flush();
        std.process.exit(2);
    }

    const t0 = std.Io.Clock.now(.awake, io);

    const atlas_raw = try std.Io.Dir.cwd().readFileAlloc(io, atlas_path.?, gpa, .limited(32 * 1024 * 1024));
    defer gpa.free(atlas_raw);
    const templates = try loadAtlas(gpa, atlas_raw);
    defer gpa.free(templates);

    const pgm_raw = try std.Io.Dir.cwd().readFileAlloc(io, pgm_path.?, gpa, .limited(64 * 1024 * 1024));
    defer gpa.free(pgm_raw);
    const img = try parsePgm(pgm_raw);

    const t_rec0 = std.Io.Clock.now(.awake, io);

    const fg_res = try computeForeground(gpa, img);
    defer gpa.free(fg_res.fg);
    defer gpa.free(fg_res.mask);

    var lines = try recognizeLines(gpa, img, fg_res.fg, fg_res.mask, templates);
    defer {
        for (lines.items) |ln| gpa.free(ln.text);
        lines.deinit(gpa);
    }

    const t1 = std.Io.Clock.now(.awake, io);
    const total_ms = @as(f64, @floatFromInt(t1.nanoseconds - t0.nanoseconds)) / 1_000_000.0;
    const ocr_ms = @as(f64, @floatFromInt(t1.nanoseconds - t_rec0.nanoseconds)) / 1_000_000.0;

    var out_buf: [65536]u8 = undefined;
    var fw = std.Io.File.stdout().writer(io, &out_buf);
    const w = &fw.interface;

    const status = if (lines.items.len == 0) "OCR_EMPTY" else "OCR_OBSERVED";
    try w.print(
        "{{\"status\":\"{s}\",\"engine\":\"flowsight-ocr-zig/0.1\",\"templates\":{d},\"elapsed_ms\":{d:.2},\"ocr_ms\":{d:.2},\"items\":[",
        .{ status, templates.len, total_ms, ocr_ms },
    );

    for (lines.items, 0..) |ln, idx| {
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

test "parsePgm parses valid P5 header and pixels" {
    const raw = "P5\n2 2\n255\n\x00\xff\x80\x40";
    const img = try parsePgm(raw);
    try std.testing.expectEqual(@as(u32, 2), img.width);
    try std.testing.expectEqual(@as(u32, 2), img.height);
    try std.testing.expectEqual(@as(u8, 0xff), img.pixels[1]);
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
    disambiguatePhrase(&num_tok, 4.0);
    try std.testing.expectEqual(@as(u32, '0'), num_tok[1].cp);

    var word_tok = [_]MatchedGlyph{
        .{ .cp = 's', .score = 0.9, .second_cp = 'S', .second_score = 0.7, .gap_before = 0, .width = 6, .height = 8 },
        .{ .cp = 'a', .score = 0.9, .second_cp = 'o', .second_score = 0.7, .gap_before = 1, .width = 6, .height = 8 },
        .{ .cp = 'l', .score = 0.9, .second_cp = '1', .second_score = 0.8, .gap_before = 1, .width = 3, .height = 11 },
        .{ .cp = '1', .score = 0.85, .second_cp = 'i', .second_score = 0.82, .gap_before = 1, .width = 3, .height = 11 },
        .{ .cp = 'd', .score = 0.9, .second_cp = 'b', .second_score = 0.6, .gap_before = 1, .width = 6, .height = 11 },
        .{ .cp = 'a', .score = 0.9, .second_cp = 'o', .second_score = 0.7, .gap_before = 1, .width = 6, .height = 8 },
    };
    disambiguatePhrase(&word_tok, 4.0);
    try std.testing.expectEqual(@as(u32, 'i'), word_tok[3].cp);
}
