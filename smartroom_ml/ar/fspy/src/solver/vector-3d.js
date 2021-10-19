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
var Vector3D = /** @class */ (function () {
    function Vector3D(x, y, z) {
        if (x === void 0) { x = 0; }
        if (y === void 0) { y = 0; }
        if (z === void 0) { z = 0; }
        this.x = x;
        this.y = y;
        this.z = z;
    }
    Vector3D.prototype.copy = function () {
        return new Vector3D(this.x, this.y, this.z);
    };
    Object.defineProperty(Vector3D.prototype, "minCoordinate", {
        get: function () {
            return Math.min(this.x, this.y, this.z);
        },
        enumerable: false,
        configurable: true
    });
    Object.defineProperty(Vector3D.prototype, "maxCoordinate", {
        get: function () {
            return Math.max(this.x, this.y, this.z);
        },
        enumerable: false,
        configurable: true
    });
    Object.defineProperty(Vector3D.prototype, "minAbsCoordinate", {
        get: function () {
            return Math.min(Math.abs(this.x), Math.abs(this.y), Math.abs(this.z));
        },
        enumerable: false,
        configurable: true
    });
    Object.defineProperty(Vector3D.prototype, "maxAbsCoordinate", {
        get: function () {
            return Math.max(Math.abs(this.x), Math.abs(this.y), Math.abs(this.z));
        },
        enumerable: false,
        configurable: true
    });
    Vector3D.prototype.add = function (other) {
        this.x += other.x;
        this.y += other.y;
        this.z += other.z;
    };
    Vector3D.prototype.added = function (other) {
        var result = this.copy();
        result.add(other);
        return result;
    };
    Vector3D.prototype.subtract = function (other) {
        this.x -= other.x;
        this.y -= other.y;
        this.z -= other.z;
    };
    Vector3D.prototype.subtracted = function (other) {
        var result = this.copy();
        result.subtract(other);
        return result;
    };
    Object.defineProperty(Vector3D.prototype, "length", {
        get: function () {
            return Math.sqrt(this.x * this.x + this.y * this.y + this.z * this.z);
        },
        enumerable: false,
        configurable: true
    });
    /**
     * Normalizes the vector and returns the length prior to normalization.
     */
    Vector3D.prototype.normalize = function () {
        var length = this.length;
        if (length > 0) {
            this.x /= length;
            this.y /= length;
            this.z /= length;
        }
        return length;
    };
    Vector3D.prototype.normalized = function () {
        var result = this.copy();
        result.normalize();
        return result;
    };
    Vector3D.prototype.negate = function () {
        this.x = -this.x;
        this.y = -this.y;
        this.z = -this.z;
    };
    Vector3D.prototype.negated = function () {
        var result = this.copy();
        result.negate();
        return result;
    };
    Vector3D.prototype.multiplyByScalar = function (scalar) {
        this.x *= scalar;
        this.y *= scalar;
        this.z *= scalar;
    };
    Vector3D.prototype.multipliedByScalar = function (scalar) {
        var result = this.copy();
        result.multiplyByScalar(scalar);
        return result;
    };
    /**
     * Dot product this * other
     * @param other
     */
    Vector3D.prototype.dot = function (other) {
        return this.x * other.x + this.y * other.y + this.z * other.z;
    };
    /**
     * Cross product this x other
     * @param other
     */
    Vector3D.prototype.cross = function (other) {
        return new Vector3D(this.y * other.z - this.z * other.y, this.z * other.x - this.x * other.z, this.x * other.y - this.y * other.x);
    };
    return Vector3D;
}());
exports.default = Vector3D;
