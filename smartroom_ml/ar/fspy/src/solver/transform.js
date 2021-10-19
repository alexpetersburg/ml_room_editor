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
var aabb_ops_1 = require("./aabb-ops");
var Transform = /** @class */ (function () {
    function Transform() {
        this.rows = [
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ];
    }
    Transform.lookatTransform = function (lookFrom, lookAt, up) {
        var forward = lookAt.subtracted(lookFrom).normalized();
        var right = forward.cross(up).normalized();
        var u = right.cross(forward);
        return Transform.fromMatrix([
            [right.x, u.x, -forward.x, lookFrom.x],
            [right.y, u.y, -forward.y, lookFrom.y],
            [right.z, u.z, -forward.z, lookFrom.z],
            [0, 0, 0, 1]
        ]).inverted();
    };
    Transform.perspectiveProjection = function (fieldOfView, near, far) {
        // http://www.songho.ca/opengl/gl_projectionmatrix.html
        var s = 1 / Math.tan(0.5 * fieldOfView);
        return Transform.fromMatrix([
            [s, 0, 0, 0],
            [0, s, 0, 0],
            [0, 0, -(far) / (far - near), -far * near / (far - near)],
            [0, 0, -1, 0]
        ]);
    };
    Transform.fromMatrix = function (matrix) {
        var transform = new Transform();
        for (var rowIndex = 0; rowIndex < 4; rowIndex++) {
            if (rowIndex >= matrix.length) {
                continue;
            }
            for (var columnIndex = 0; columnIndex < 4; columnIndex++) {
                if (columnIndex >= matrix[rowIndex].length) {
                    break;
                }
                transform.rows[rowIndex][columnIndex] = matrix[rowIndex][columnIndex];
            }
        }
        return transform;
    };
    Transform.rotation = function (angle, xAxis, yAxis, zAxis) {
        if (xAxis === void 0) { xAxis = 0; }
        if (yAxis === void 0) { yAxis = 0; }
        if (zAxis === void 0) { zAxis = 1; }
        // https://en.wikipedia.org/wiki/Rotation_matrix
        var axis = new vector_3d_1.default(xAxis, yAxis, zAxis).normalized();
        var transform = new Transform();
        transform.rows[0][0] = Math.cos(angle) + axis.x * axis.x * (1 - Math.cos(angle));
        transform.rows[0][1] = axis.x * axis.y * (1 - Math.cos(angle)) - axis.z * Math.sin(angle);
        transform.rows[0][2] = axis.x * axis.z * (1 - Math.cos(angle)) + axis.y * Math.sin(angle);
        transform.rows[1][0] = axis.y * axis.x * (1 - Math.cos(angle)) + axis.z * Math.sin(angle);
        transform.rows[1][1] = Math.cos(angle) + axis.y * axis.y * (1 - Math.cos(angle));
        transform.rows[1][2] = axis.y * axis.z * (1 - Math.cos(angle)) - axis.x * Math.sin(angle);
        transform.rows[2][0] = axis.z * axis.x * (1 - Math.cos(angle)) - axis.y * Math.sin(angle);
        transform.rows[2][1] = axis.z * axis.y * (1 - Math.cos(angle)) + axis.x * Math.sin(angle);
        transform.rows[2][2] = Math.cos(angle) + axis.z * axis.z * (1 - Math.cos(angle));
        return transform;
    };
    Transform.scale = function (sx, sy, sz) {
        if (sz === void 0) { sz = 0; }
        var transform = new Transform();
        transform.rows[0][0] = sx;
        transform.rows[1][1] = sy;
        transform.rows[2][2] = sz;
        return transform;
    };
    Transform.translation = function (dx, dy, dz) {
        if (dz === void 0) { dz = 0; }
        var transform = new Transform();
        transform.rows[0][3] = dx;
        transform.rows[1][3] = dy;
        transform.rows[2][3] = dz;
        return transform;
    };
    Transform.skew2D = function (xAngle, yAngle) {
        var transform = new Transform();
        transform.rows[0][1] = Math.tan(xAngle);
        transform.rows[1][0] = Math.tan(yAngle);
        return transform;
    };
    /**
     * Concatenates a number of transforms. [A, B, C] => ABC
     * @param transforms [a, b, c]
     */
    Transform.concatenate = function (transforms) {
        var result = new Transform();
        for (var i = 0; i < transforms.length; i++) {
            var transform = transforms[transforms.length - 1 - i];
            result.leftMultiply(transform);
        }
        return result;
    };
    Transform.prototype.copy = function () {
        var copy = new Transform();
        for (var rowIndex = 0; rowIndex < 4; rowIndex++) {
            for (var columnIndex = 0; columnIndex < 4; columnIndex++) {
                copy.matrix[rowIndex][columnIndex] = this.rows[rowIndex][columnIndex];
            }
        }
        return copy;
    };
    Object.defineProperty(Transform.prototype, "matrix", {
        get: function () {
            return this.rows;
        },
        enumerable: false,
        configurable: true
    });
    Object.defineProperty(Transform.prototype, "isIdentity", {
        get: function () {
            for (var rowIndex = 0; rowIndex < this.rows.length; rowIndex++) {
                for (var columnIndex = 0; columnIndex < this.rows.length; columnIndex++) {
                    var expectedValue = rowIndex == columnIndex ? 1 : 0;
                    if (this.rows[rowIndex][columnIndex] != expectedValue) {
                        return false;
                    }
                }
            }
            return true;
        },
        enumerable: false,
        configurable: true
    });
    Object.defineProperty(Transform.prototype, "isDiagonal", {
        get: function () {
            for (var rowIndex = 0; rowIndex < this.rows.length; rowIndex++) {
                for (var columnIndex = 0; columnIndex < this.rows.length; columnIndex++) {
                    if (rowIndex == columnIndex) {
                        if (this.rows[rowIndex][columnIndex] == 0) {
                            return false;
                        }
                    }
                    else {
                        if (this.rows[rowIndex][columnIndex] != 0) {
                            return false;
                        }
                    }
                }
            }
            return true;
        },
        enumerable: false,
        configurable: true
    });
    Object.defineProperty(Transform.prototype, "determinant", {
        get: function () {
            // http://www.euclideanspace.com/maths/algebra/matrix/functions/determinant/fourD/index.htm
            var m = this.matrix;
            var m00 = m[0][0];
            var m01 = m[0][1];
            var m02 = m[0][2];
            var m03 = m[0][3];
            var m10 = m[1][0];
            var m11 = m[1][1];
            var m12 = m[1][2];
            var m13 = m[1][3];
            var m20 = m[2][0];
            var m21 = m[2][1];
            var m22 = m[2][2];
            var m23 = m[2][3];
            var m30 = m[3][0];
            var m31 = m[3][1];
            var m32 = m[3][2];
            var m33 = m[3][3];
            var result = m03 * m12 * m21 * m30 - m02 * m13 * m21 * m30 - m03 * m11 * m22 * m30 + m01 * m13 * m22 * m30 +
                m02 * m11 * m23 * m30 - m01 * m12 * m23 * m30 - m03 * m12 * m20 * m31 + m02 * m13 * m20 * m31 +
                m03 * m10 * m22 * m31 - m00 * m13 * m22 * m31 - m02 * m10 * m23 * m31 + m00 * m12 * m23 * m31 +
                m03 * m11 * m20 * m32 - m01 * m13 * m20 * m32 - m03 * m10 * m21 * m32 + m00 * m13 * m21 * m32 +
                m01 * m10 * m23 * m32 - m00 * m11 * m23 * m32 - m02 * m11 * m20 * m33 + m01 * m12 * m20 * m33 +
                m02 * m10 * m21 * m33 - m00 * m12 * m21 * m33 - m01 * m10 * m22 * m33 + m00 * m11 * m22 * m33;
            return result;
        },
        enumerable: false,
        configurable: true
    });
    Transform.prototype.invert = function () {
        // http://www.euclideanspace.com/maths/algebra/matrix/functions/inverse/fourD/index.htm
        var m = this.matrix;
        var m00 = m[0][0];
        var m01 = m[0][1];
        var m02 = m[0][2];
        var m03 = m[0][3];
        var m10 = m[1][0];
        var m11 = m[1][1];
        var m12 = m[1][2];
        var m13 = m[1][3];
        var m20 = m[2][0];
        var m21 = m[2][1];
        var m22 = m[2][2];
        var m23 = m[2][3];
        var m30 = m[3][0];
        var m31 = m[3][1];
        var m32 = m[3][2];
        var m33 = m[3][3];
        var s = 1 / this.determinant;
        this.rows[0][0] = m12 * m23 * m31 - m13 * m22 * m31 + m13 * m21 * m32 - m11 * m23 * m32 - m12 * m21 * m33 + m11 * m22 * m33;
        this.rows[0][1] = m03 * m22 * m31 - m02 * m23 * m31 - m03 * m21 * m32 + m01 * m23 * m32 + m02 * m21 * m33 - m01 * m22 * m33;
        this.rows[0][2] = m02 * m13 * m31 - m03 * m12 * m31 + m03 * m11 * m32 - m01 * m13 * m32 - m02 * m11 * m33 + m01 * m12 * m33;
        this.rows[0][3] = m03 * m12 * m21 - m02 * m13 * m21 - m03 * m11 * m22 + m01 * m13 * m22 + m02 * m11 * m23 - m01 * m12 * m23;
        this.rows[1][0] = m13 * m22 * m30 - m12 * m23 * m30 - m13 * m20 * m32 + m10 * m23 * m32 + m12 * m20 * m33 - m10 * m22 * m33;
        this.rows[1][1] = m02 * m23 * m30 - m03 * m22 * m30 + m03 * m20 * m32 - m00 * m23 * m32 - m02 * m20 * m33 + m00 * m22 * m33;
        this.rows[1][2] = m03 * m12 * m30 - m02 * m13 * m30 - m03 * m10 * m32 + m00 * m13 * m32 + m02 * m10 * m33 - m00 * m12 * m33;
        this.rows[1][3] = m02 * m13 * m20 - m03 * m12 * m20 + m03 * m10 * m22 - m00 * m13 * m22 - m02 * m10 * m23 + m00 * m12 * m23;
        this.rows[2][0] = m11 * m23 * m30 - m13 * m21 * m30 + m13 * m20 * m31 - m10 * m23 * m31 - m11 * m20 * m33 + m10 * m21 * m33;
        this.rows[2][1] = m03 * m21 * m30 - m01 * m23 * m30 - m03 * m20 * m31 + m00 * m23 * m31 + m01 * m20 * m33 - m00 * m21 * m33;
        this.rows[2][2] = m01 * m13 * m30 - m03 * m11 * m30 + m03 * m10 * m31 - m00 * m13 * m31 - m01 * m10 * m33 + m00 * m11 * m33;
        this.rows[2][3] = m03 * m11 * m20 - m01 * m13 * m20 - m03 * m10 * m21 + m00 * m13 * m21 + m01 * m10 * m23 - m00 * m11 * m23;
        this.rows[3][0] = m12 * m21 * m30 - m11 * m22 * m30 - m12 * m20 * m31 + m10 * m22 * m31 + m11 * m20 * m32 - m10 * m21 * m32;
        this.rows[3][1] = m01 * m22 * m30 - m02 * m21 * m30 + m02 * m20 * m31 - m00 * m22 * m31 - m01 * m20 * m32 + m00 * m21 * m32;
        this.rows[3][2] = m02 * m11 * m30 - m01 * m12 * m30 - m02 * m10 * m31 + m00 * m12 * m31 + m01 * m10 * m32 - m00 * m11 * m32;
        this.rows[3][3] = m01 * m12 * m20 - m02 * m11 * m20 + m02 * m10 * m21 - m00 * m12 * m21 - m01 * m10 * m22 + m00 * m11 * m22;
        for (var rowIndex = 0; rowIndex < 4; rowIndex++) {
            for (var columnIndex = 0; columnIndex < 4; columnIndex++) {
                this.rows[rowIndex][columnIndex] *= s;
            }
        }
    };
    Transform.prototype.inverted = function () {
        var result = this.copy();
        result.invert();
        return result;
    };
    Transform.prototype.transpose = function () {
        for (var rowIndex = 0; rowIndex < this.rows.length; rowIndex++) {
            for (var columnIndex = 0; columnIndex < rowIndex; columnIndex++) {
                var lowerValue = this.rows[rowIndex][columnIndex];
                var upperValue = this.rows[columnIndex][rowIndex];
                this.rows[rowIndex][columnIndex] = upperValue;
                this.rows[columnIndex][rowIndex] = lowerValue;
            }
        }
    };
    Transform.prototype.transposed = function () {
        var result = this.copy();
        result.transpose();
        return result;
    };
    Transform.prototype.transformVector = function (vector, perspectiveDivide, targetRect) {
        if (perspectiveDivide === void 0) { perspectiveDivide = false; }
        if (targetRect === void 0) { targetRect = null; }
        var result = this.transform([vector.x, vector.y, vector.z, 1]);
        vector.x = result[0];
        vector.y = result[1];
        vector.z = result[2];
        if (perspectiveDivide) {
            vector.x /= result[3];
            vector.y /= result[3];
            vector.z /= result[3];
        }
        if (targetRect) {
            vector.x = 0.5 * aabb_ops_1.default.width(targetRect) * vector.x + aabb_ops_1.default.midPoint(targetRect).x;
            vector.y = -0.5 * aabb_ops_1.default.height(targetRect) * vector.y + aabb_ops_1.default.midPoint(targetRect).y;
        }
    };
    Transform.prototype.transformedVector = function (vector, perspectiveDivide) {
        if (perspectiveDivide === void 0) { perspectiveDivide = false; }
        var copy = vector.copy();
        this.transformVector(copy, perspectiveDivide);
        return copy;
    };
    Transform.prototype.transformVectors = function (vectors) {
        for (var _i = 0, vectors_1 = vectors; _i < vectors_1.length; _i++) {
            var vector = vectors_1[_i];
            this.transformVector(vector);
        }
    };
    Transform.prototype.transformedVectors = function (vectors) {
        var result = [];
        for (var _i = 0, vectors_2 = vectors; _i < vectors_2.length; _i++) {
            var vector = vectors_2[_i];
            result.push(this.transformedVector(vector));
        }
        return result;
    };
    Transform.prototype.transform2DPoint = function (point) {
        var result = this.transform([point.x, point.y, 0, 1]);
        point.x = result[0];
        point.y = result[1];
    };
    Transform.prototype.transform2DPoints = function (points) {
        for (var _i = 0, points_1 = points; _i < points_1.length; _i++) {
            var point = points_1[_i];
            this.transform2DPoint(point);
        }
    };
    Transform.prototype.transformed2DPoint = function (point) {
        var result = this.transform([point.x, point.y, 0, 1]);
        return {
            x: result[0],
            y: result[1]
        };
    };
    Transform.prototype.transformed2DPoints = function (points) {
        var result = [];
        for (var _i = 0, points_2 = points; _i < points_2.length; _i++) {
            var point = points_2[_i];
            result.push(this.transformed2DPoint(point));
        }
        return result;
    };
    Transform.prototype.equals = function (transform, epsilon) {
        if (epsilon === void 0) { epsilon = 0; }
        for (var rowIndex = 0; rowIndex < this.rows.length; rowIndex++) {
            for (var columnIndex = 0; columnIndex < this.rows[rowIndex].length; columnIndex++) {
                var thisValue = this.rows[rowIndex][columnIndex];
                var otherValue = transform.rows[rowIndex][columnIndex];
                if (epsilon == 0) {
                    if (thisValue != otherValue) {
                        return false;
                    }
                }
                else {
                    if (Math.abs(thisValue - otherValue) > epsilon) {
                        return false;
                    }
                }
            }
        }
        return true;
    };
    Transform.prototype.translate = function (dx, dy, dz) {
        if (dz === void 0) { dz = 0; }
        this.rows[0][3] += dx;
        this.rows[1][3] += dy;
        this.rows[2][3] += dz;
    };
    Transform.prototype.translated = function (dx, dy, dz) {
        if (dz === void 0) { dz = 0; }
        var transform = this.copy();
        transform.translate(dx, dy, dz);
        return transform;
    };
    Transform.prototype.scale = function (sx, sy, sz) {
        if (sz === void 0) { sz = 0; }
        this.rows[0][0] *= sx;
        this.rows[1][1] *= sy;
        this.rows[2][2] *= sz;
    };
    Transform.prototype.scaled = function (sx, sy, sz) {
        if (sz === void 0) { sz = 0; }
        var transform = this.copy();
        transform.scale(sx, sy, sz);
        return transform;
    };
    Transform.prototype.scaleUniform = function (s) {
        this.rows[0][0] *= s;
        this.rows[1][1] *= s;
        this.rows[2][2] *= s;
    };
    Transform.prototype.scaledUniform = function (s) {
        var transform = this.copy();
        transform.scale(s, s, s);
        return transform;
    };
    Transform.prototype.rotate = function (angle, xAxis, yAxis, zAxis) {
        if (xAxis === void 0) { xAxis = 0; }
        if (yAxis === void 0) { yAxis = 0; }
        if (zAxis === void 0) { zAxis = 1; }
        var rotationTransform = Transform.rotation(angle, xAxis, yAxis, zAxis);
        this.leftMultiply(rotationTransform);
    };
    Transform.prototype.rotated = function (angle, xAxis, yAxis, zAxis) {
        if (xAxis === void 0) { xAxis = 0; }
        if (yAxis === void 0) { yAxis = 0; }
        if (zAxis === void 0) { zAxis = 1; }
        var transform = this.copy();
        transform.rotate(angle, xAxis, yAxis, zAxis);
        return transform;
    };
    /**
     * result = transform * this
     * @param transform
     */
    Transform.prototype.leftMultiply = function (transform) {
        var result = [
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0]
        ];
        for (var rowIndex = 0; rowIndex < 4; rowIndex++) {
            for (var columnIndex = 0; columnIndex < 4; columnIndex++) {
                var sum = 0;
                for (var srcRow = 0; srcRow < 4; srcRow++) {
                    sum += this.rows[srcRow][columnIndex] * transform.rows[rowIndex][srcRow];
                }
                result[rowIndex][columnIndex] = sum;
            }
        }
        this.rows = result;
    };
    Transform.prototype.leftMultiplied = function (transform) {
        var result = this.copy();
        result.leftMultiply(transform);
        return result;
    };
    Object.defineProperty(Transform.prototype, "svgString", {
        get: function () {
            // https://developer.mozilla.org/en-US/docs/Web/SVG/Attribute/transform
            // matrix(<a> <b> <c> <d> <e> <f>)
            // a c e
            // b d f
            // 0 0 1
            //
            // a c * e
            // b d * f
            // * * * *
            // 0 0 0 1
            var a = this.rows[0][0];
            var b = this.rows[1][0];
            var c = this.rows[0][1];
            var d = this.rows[1][1];
            var e = this.rows[0][3];
            var f = this.rows[1][3];
            return 'matrix(' + a + ' ' + b + ' ' + c + ' ' + d + ' ' + e + ' ' + f + ')';
        },
        enumerable: false,
        configurable: true
    });
    Transform.prototype.transform = function (vector) {
        var result = [];
        for (var rowIndex = 0; rowIndex < 4; rowIndex++) {
            var coord = 0;
            for (var columnIndex = 0; columnIndex < 4; columnIndex++) {
                coord += this.rows[rowIndex][columnIndex] * vector[columnIndex];
            }
            result.push(coord);
        }
        return result;
    };
    return Transform;
}());
exports.default = Transform;
