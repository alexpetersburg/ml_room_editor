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
var AABBOps = /** @class */ (function () {
    function AABBOps() {
    }
    AABBOps.width = function (aabb) {
        return Math.abs(aabb.xMax - aabb.xMin);
    };
    AABBOps.height = function (aabb) {
        return Math.abs(aabb.yMax - aabb.yMin);
    };
    AABBOps.maxDimension = function (aabb) {
        return Math.max(this.width(aabb), this.height(aabb));
    };
    AABBOps.minDimension = function (aabb) {
        return Math.min(this.width(aabb), this.height(aabb));
    };
    AABBOps.aspectRatio = function (aabb) {
        var height = this.height(aabb);
        if (height == 0) {
            return 0;
        }
        return this.width(aabb) / height;
    };
    AABBOps.relativePosition = function (aabb, xRelative, yRelative) {
        return {
            x: aabb.xMin + xRelative * this.width(aabb),
            y: aabb.yMin + yRelative * this.height(aabb)
        };
    };
    AABBOps.midPoint = function (aabb) {
        return this.relativePosition(aabb, 0.5, 0.5);
    };
    AABBOps.boundingBox = function (points, padding) {
        if (padding === void 0) { padding = 0; }
        if (points.length == 0) {
            return { xMin: 0, yMin: 0, xMax: 0, yMax: 0 };
        }
        var xMin = points[0].x;
        var xMax = xMin;
        var yMin = points[0].y;
        var yMax = yMin;
        for (var _i = 0, points_1 = points; _i < points_1.length; _i++) {
            var point = points_1[_i];
            if (point.x > xMax) {
                xMax = point.x;
            }
            else if (point.x < xMin) {
                xMin = point.x;
            }
            if (point.y > yMax) {
                yMax = point.y;
            }
            else if (point.y < yMin) {
                yMin = point.y;
            }
        }
        return {
            xMin: xMin - padding,
            yMin: yMin - padding,
            xMax: xMax + padding,
            yMax: yMax + padding
        };
    };
    /**
     * Returns the largest rect with a given aspect ratio
     * that fits inside this rect.
     * @param aspectRatio
     * @param relativeOffset
     */
    AABBOps.maximumInteriorAABB = function (aabb, aspectRatio, relativeOffset) {
        if (relativeOffset === void 0) { relativeOffset = 0.5; }
        var width = this.width(aabb);
        var height = this.height(aabb);
        if (this.aspectRatio(aabb) < aspectRatio) {
            // The rect to fit is wider than the containing rect. Cap width.
            height = this.width(aabb) / aspectRatio;
        }
        else {
            // The rect to fit is taller than the containing rect. Cap height.
            width = this.height(aabb) * aspectRatio;
        }
        var xMin = aabb.xMin + relativeOffset * (this.width(aabb) - width);
        var yMin = aabb.yMin + relativeOffset * (this.height(aabb) - height);
        return {
            xMin: xMin,
            yMin: yMin,
            xMax: xMin + width,
            yMax: yMin + height
        };
    };
    AABBOps.containsPoint = function (aabb, point) {
        var xInside = point.x >= aabb.xMin && point.x <= aabb.xMax;
        var yInside = point.y >= aabb.yMin && point.y <= aabb.yMax;
        return xInside && yInside;
    };
    AABBOps.overlaps = function (aabb1, aabb2, padding) {
        if (padding === void 0) { padding = 0; }
        var xSeparation = aabb1.xMax + padding < aabb2.xMin || aabb1.xMin - padding > aabb2.xMax;
        var ySeparation = aabb1.yMax + padding < aabb2.yMin || aabb1.yMin - padding > aabb2.yMax;
        return !xSeparation && !ySeparation;
    };
    /**
     * Returns true if aabb1 contains aabb2, false otherwise
     * @param aabb1
     * @param aabb2
     * @param padding
     */
    AABBOps.contains = function (aabb1, aabb2) {
        var containsX = aabb1.xMin <= aabb2.xMin && aabb1.xMax >= aabb2.xMax;
        var containsY = aabb1.yMin <= aabb2.yMin && aabb1.yMax >= aabb2.yMax;
        return containsX && containsY;
    };
    return AABBOps;
}());
exports.default = AABBOps;
