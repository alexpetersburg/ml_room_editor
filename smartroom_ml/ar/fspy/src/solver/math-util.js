"use strict";
/**
 * fSpy
 * Copyright (c) 2020 - Per Gantelius
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <http://www.gnu.org/licenses/>.
 */
Object.defineProperty(exports, "__esModule", { value: true });
var vector_3d_1 = require("./vector-3d");
var transform_1 = require("./transform");
var MathUtil = /** @class */ (function () {
    function MathUtil() {
    }
    MathUtil.normalized = function (vector) {
        var l = this.distance({ x: 0, y: 0 }, vector);
        if (l != 0) {
            return {
                x: vector.x / l,
                y: vector.y / l
            };
        }
        // zero length vector. really undefined, but return something anyway
        return {
            x: 0,
            y: 0
        };
    };
    MathUtil.dot = function (a, b) {
        return a.x * b.x + a.y * b.y;
    };
    MathUtil.difference = function (a, b) {
        return {
            x: a.x - b.x,
            y: a.y - b.y
        };
    };
    MathUtil.distance = function (a, b) {
        var dx = a.x - b.x;
        var dy = a.y - b.y;
        return Math.sqrt(dx * dx + dy * dy);
    };
    MathUtil.lineSegmentMidpoint = function (segment) {
        return {
            x: 0.5 * (segment[0].x + segment[1].x),
            y: 0.5 * (segment[0].y + segment[1].y)
        };
    };
    MathUtil.lineIntersection = function (line1, line2) {
        var d1 = this.distance(line1[0], line1[1]);
        var d2 = this.distance(line2[0], line2[1]);
        var epsilon = 1e-8;
        if (Math.abs(d1) < epsilon || Math.abs(d2) < epsilon) {
            return null;
        }
        // https://en.wikipedia.org/wiki/Line–line_intersection
        var x1 = line1[0].x;
        var y1 = line1[0].y;
        var x2 = line1[1].x;
        var y2 = line1[1].y;
        var x3 = line2[0].x;
        var y3 = line2[0].y;
        var x4 = line2[1].x;
        var y4 = line2[1].y;
        var denominator = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4);
        if (Math.abs(denominator) < epsilon) {
            return null;
        }
        return {
            x: ((x1 * y2 - y1 * x2) * (x3 - x4) - (x1 - x2) * (x3 * y4 - y3 * x4)) / denominator,
            y: ((x1 * y2 - y1 * x2) * (y3 - y4) - (y1 - y2) * (x3 * y4 - y3 * x4)) / denominator
        };
    };
    MathUtil.triangleOrthoCenter = function (k, l, m) {
        var a = k.x;
        var b = k.y;
        var c = l.x;
        var d = l.y;
        var e = m.x;
        var f = m.y;
        var N = b * c + d * e + f * a - c * f - b * e - a * d;
        var x = ((d - f) * b * b + (f - b) * d * d + (b - d) * f * f + a * b * (c - e) + c * d * (e - a) + e * f * (a - c)) / N;
        var y = ((e - c) * a * a + (a - e) * c * c + (c - a) * e * e + a * b * (f - d) + c * d * (b - f) + e * f * (d - b)) / N;
        return {
            x: x, y: y
        };
    };
    MathUtil.thirdTriangleVertex = function (firstVertex, secondVertex, orthocenter) {
        var a = firstVertex;
        var b = secondVertex;
        var o = orthocenter;
        // compute p, the orthogonal projection of the orthocenter onto the line through a and b
        var aToB = this.normalized({ x: b.x - a.x, y: b.y - a.y });
        var proj = this.dot(aToB, this.difference(o, a));
        var p = {
            x: a.x + proj * aToB.x,
            y: a.y + proj * aToB.y
        };
        // the vertex c can be expressed as p + hn, where n is orthogonal to ab.
        var n = { x: aToB.y, y: -aToB.x };
        var h = this.dot(this.difference(a, p), this.difference(o, b)) / (this.dot(n, this.difference(o, b)));
        return {
            x: p.x + h * n.x,
            y: p.y + h * n.y
        };
    };
    MathUtil.linePlaneIntersection = function (p0, p1, p2, la, lb) {
        // https://en.wikipedia.org/wiki/Line–plane_intersection
        var p01 = p1.subtracted(p0);
        var p02 = p2.subtracted(p0);
        var lab = lb.subtracted(la);
        var numerator = (p01.cross(p02)).dot(la.subtracted(p0));
        var denominator = -(lab.dot(p01.cross(p02)));
        var t = numerator / denominator;
        return new vector_3d_1.default(la.x + t * lab.x, la.y + t * lab.y, la.z + t * lab.z);
    };
    MathUtil.shortestLineSegmentBetweenLines = function (p1, p2, p3, p4) {
        // TODO: gracefully handle parallel lines
        // http://paulbourke.net/geometry/pointlineplane/
        function d(m, n, o, p) {
            // dmnop = (xm - xn)(xo - xp) + (ym - yn)(yo - yp) + (zm - zn)(zo - zp)
            var allPoints = [p1, p2, p3, p4];
            var pm = allPoints[m - 1];
            var pn = allPoints[n - 1];
            var po = allPoints[o - 1];
            var pp = allPoints[p - 1];
            return (pm.x - pn.x) * (po.x - pp.x) + (pm.y - pn.y) * (po.y - pp.y) + (pm.z - pn.z) * (po.z - pp.z);
        }
        var muaNumerator = d(1, 3, 4, 3) * d(4, 3, 2, 1) - d(1, 3, 2, 1) * d(4, 3, 4, 3);
        var muaDenominator = d(2, 1, 2, 1) * d(4, 3, 4, 3) - d(4, 3, 2, 1) * d(4, 3, 2, 1);
        var mua = muaNumerator / muaDenominator;
        var mub = (d(1, 3, 4, 3) + mua * d(4, 3, 2, 1)) / d(4, 3, 4, 3);
        return [
            new vector_3d_1.default(p1.x + mua * (p2.x - p1.x), p1.y + mua * (p2.y - p1.y), p1.z + mua * (p2.z - p1.z)),
            new vector_3d_1.default(p3.x + mub * (p4.x - p3.x), p3.y + mub * (p4.y - p3.y), p3.z + mub * (p4.z - p3.z))
        ];
    };
    MathUtil.perspectiveUnproject = function (point, viewTransform, principalPoint, horizontalFieldOfView) {
        var transform = this.modelViewProjection(viewTransform, principalPoint, horizontalFieldOfView).inverted();
        return transform.transformedVector(point, true);
    };
    MathUtil.perspectiveProject = function (point, viewTransform, principalPoint, horizontalFieldOfView) {
        var projected = this.modelViewProjection(viewTransform, principalPoint, horizontalFieldOfView).transformedVector(point, true);
        return projected;
    };
    MathUtil.pointsAreOnTheSameSideOfLine = function (l1, l2, p1, p2) {
        var lineDirection = {
            x: l2.x - l1.x,
            y: l2.y - l1.y
        };
        var lineNormal = {
            x: lineDirection.y,
            y: -lineDirection.x
        };
        var l1ToP1 = {
            x: p1.x - l1.x,
            y: p1.y - l1.y
        };
        var l1ToP2 = {
            x: p2.x - l1.x,
            y: p2.y - l1.y
        };
        var dot1 = l1ToP1.x * lineNormal.x + l1ToP1.y * lineNormal.y;
        var dot2 = l1ToP2.x * lineNormal.x + l1ToP2.y * lineNormal.y;
        return dot1 * dot2 > 0;
    };
    MathUtil.matrixToAxisAngle = function (transform) {
        // http://www.euclideanspace.com/maths/geometry/rotations/conversions/matrixToAngle/
        var m00 = transform.matrix[0][0];
        var m01 = transform.matrix[0][1];
        var m02 = transform.matrix[0][2];
        var m10 = transform.matrix[1][0];
        var m11 = transform.matrix[1][1];
        var m12 = transform.matrix[1][2];
        var m20 = transform.matrix[2][0];
        var m21 = transform.matrix[2][1];
        var m22 = transform.matrix[2][2];
        var x = (m21 - m12) / Math.sqrt((m21 - m12) * (m21 - m12) + (m02 - m20) * (m02 - m20) + (m10 - m01) * (m10 - m01));
        var y = (m02 - m20) / Math.sqrt((m21 - m12) * (m21 - m12) + (m02 - m20) * (m02 - m20) + (m10 - m01) * (m10 - m01));
        var z = (m10 - m01) / Math.sqrt((m21 - m12) * (m21 - m12) + (m02 - m20) * (m02 - m20) + (m10 - m01) * (m10 - m01));
        var angle = Math.acos((m00 + m11 + m22 - 1) / 2);
        return [x, y, z, angle];
    };
    MathUtil.matrixToQuaternion = function (transform) {
        // http://www.euclideanspace.com/maths/geometry/rotations/conversions/matrixToQuaternion/
        var m00 = transform.matrix[0][0];
        var m01 = transform.matrix[0][1];
        var m02 = transform.matrix[0][2];
        var m10 = transform.matrix[1][0];
        var m11 = transform.matrix[1][1];
        var m12 = transform.matrix[1][2];
        var m20 = transform.matrix[2][0];
        var m21 = transform.matrix[2][1];
        var m22 = transform.matrix[2][2];
        var qw = Math.sqrt(1 + m00 + m11 + m22) / 2;
        var qx = (m21 - m12) / (4 * qw);
        var qy = (m02 - m20) / (4 * qw);
        var qz = (m10 - m01) / (4 * qw);
        return [qx, qy, qz, qw];
    };
    MathUtil.modelViewProjection = function (viewTransform, principalPoint, horizontalFieldOfView) {
        var s = 1 / Math.tan(0.5 * horizontalFieldOfView);
        var n = 0.01;
        var f = 10;
        var projectionTransform = transform_1.default.fromMatrix([
            [s, 0, -principalPoint.x, 0],
            [0, s, -principalPoint.y, 0],
            [0, 0, -(f + n) / (f - n), -2 * f * n / (f - n)],
            [0, 0, -1, 0]
        ]);
        return viewTransform.leftMultiplied(projectionTransform);
    };
    return MathUtil;
}());
exports.default = MathUtil;
