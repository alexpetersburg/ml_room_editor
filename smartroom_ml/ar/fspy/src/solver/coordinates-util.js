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
exports.ImageCoordinateFrame = void 0;
/**
 * An enumeration  of coordinate frames used to specify
 * points in an image
 */
var ImageCoordinateFrame;
(function (ImageCoordinateFrame) {
    /**
     * x = 0 and x = 1 correspond to the left and right edges respectively.
     * y = 0 and y = 1 correspond to the top and bottom edges respectively,
     * The positive y direction is down.
     */
    ImageCoordinateFrame[ImageCoordinateFrame["Relative"] = 0] = "Relative";
    /**
     * x = 0 and x = w correspond to the left and right edges respectively,
     * where w is the image width in pixels.
     * y = 0 and y = h correspond to the top and bottom edges respectively,
     * where h is the image height in pixels.
     * The positive y direction is down.
     */
    ImageCoordinateFrame[ImageCoordinateFrame["Absolute"] = 1] = "Absolute";
    /**
     * A normalized coordinate frame taking into account the aspect ratio r of
     * the image. 0,0 corresponds to the image midoint.
     * For a tall image, x min = -r, x max = r, y min = -1 and y max = 1
     * For a wide image, x min = -1, x max = 1, y min = -1/aspect and y max = 1/aspect
     * The positive y direction is up
     */
    ImageCoordinateFrame[ImageCoordinateFrame["ImagePlane"] = 2] = "ImagePlane";
})(ImageCoordinateFrame = exports.ImageCoordinateFrame || (exports.ImageCoordinateFrame = {}));
var CoordinatesUtil = /** @class */ (function () {
    function CoordinatesUtil() {
    }
    /**
     * Convert a point from one image coordinate frame to another
     * @param point The point to convert
     * @param sourceFrame The coordinate frame of the input point
     * @param targetFrame The frame to convert to
     * @param imageWidth The image width in pixels
     * @param imageHeight The image height in pixels
     */
    CoordinatesUtil.convert = function (point, sourceFrame, targetFrame, imageWidth, imageHeight) {
        switch (sourceFrame) {
            case ImageCoordinateFrame.Absolute: {
                switch (targetFrame) {
                    case ImageCoordinateFrame.Absolute:
                        return point;
                    case ImageCoordinateFrame.ImagePlane:
                        var relativePoint = this.absoluteToRelative(point, imageWidth, imageHeight);
                        return this.relativeToImagePlane(relativePoint, imageWidth, imageHeight);
                    case ImageCoordinateFrame.Relative:
                        return this.absoluteToRelative(point, imageWidth, imageHeight);
                }
                break;
            }
            case ImageCoordinateFrame.ImagePlane: {
                switch (targetFrame) {
                    case ImageCoordinateFrame.Absolute:
                        var relativePoint = this.imagePlaneToRelative(point, imageWidth, imageHeight);
                        return this.relativeToAbsolute(relativePoint, imageWidth, imageHeight);
                    case ImageCoordinateFrame.ImagePlane:
                        return point;
                    case ImageCoordinateFrame.Relative:
                        return this.imagePlaneToRelative(point, imageWidth, imageHeight);
                }
                break;
            }
            case ImageCoordinateFrame.Relative: {
                switch (targetFrame) {
                    case ImageCoordinateFrame.Absolute:
                        return this.relativeToAbsolute(point, imageWidth, imageHeight);
                    case ImageCoordinateFrame.ImagePlane:
                        return this.relativeToImagePlane(point, imageWidth, imageHeight);
                    case ImageCoordinateFrame.Relative:
                        return point;
                }
                break;
            }
        }
        throw new Error('Should not end up here');
    };
    CoordinatesUtil.absoluteToRelative = function (point, imageWidth, imageHeight) {
        return {
            x: point.x / imageWidth,
            y: point.y / imageHeight
        };
    };
    CoordinatesUtil.relativeToAbsolute = function (point, imageWidth, imageHeight) {
        return {
            x: point.x * imageWidth,
            y: point.y * imageHeight
        };
    };
    CoordinatesUtil.relativeToImagePlane = function (point, imageWidth, imageHeight) {
        var aspectRatio = imageWidth / imageHeight;
        if (aspectRatio <= 1) {
            // tall image. [0, 1] x [0, 1] => [-aspect, aspect] x [-1, 1]
            return {
                x: (-1 + 2 * point.x) * aspectRatio,
                y: 1 - 2 * point.y
            };
        }
        else {
            // wide image. [0, 1] x [0, 1] => [-1, 1] x [-1 / aspect, 1 / aspect]
            return {
                x: -1 + 2 * point.x,
                y: (1 - 2 * point.y) / aspectRatio
            };
        }
    };
    CoordinatesUtil.imagePlaneToRelative = function (point, imageWidth, imageHeight) {
        var imageAspect = imageWidth / imageHeight;
        if (imageAspect <= 1) {
            // tall image. [-aspect, aspect] x [-1, 1] => [0, 1] x [0, 1]
            return {
                x: 0.5 * (point.x / imageAspect + 1),
                y: 0.5 * (-point.y + 1)
            };
        }
        else {
            // wide image. [-1, 1] x [-1 / aspect, 1 / aspect] => [0, 1] x [0, 1]
            return {
                x: 0.5 * (point.x + 1),
                y: 0.5 * (-point.y * imageAspect + 1)
            };
        }
    };
    return CoordinatesUtil;
}());
exports.default = CoordinatesUtil;
