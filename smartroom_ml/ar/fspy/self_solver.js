"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
var solve_1 = require("./src/solver/solve");
var math_util_1 = require("./src/solver/math-util");
var calibration_settings_1 = require("./src/defaults/calibration-settings");
var coordinates_util_1 = require("./src/solver/coordinates-util");
// var principalPoint = math_util_1.default.triangleOrthoCenter(coordinates_util_1.default.convert(vp1, coordinates_util_1.ImageCoordinateFrame.Absolute, coordinates_util_1.ImageCoordinateFrame.Relative, imageWidth, imageHeight), coordinates_util_1.default.convert(vp2, coordinates_util_1.ImageCoordinateFrame.Absolute, coordinates_util_1.ImageCoordinateFrame.Relative, imageWidth, imageHeight), coordinates_util_1.default.convert(vp3, coordinates_util_1.ImageCoordinateFrame.Absolute, coordinates_util_1.ImageCoordinateFrame.Relative, imageWidth, imageHeight));
let [,, imageWidth, imageHeight, x1, y1, x2, y2] = process.argv

imageWidth = parseFloat(imageWidth)
imageHeight = parseFloat(imageHeight)
x1 = parseFloat(x1)
y1 = parseFloat(y1)
x2 = parseFloat(x2)
y2 = parseFloat(y2)
let principalPoint = {x: 0, y: 0};
var vp1 = { x: x1, y: y1 };
var vp2 = { x: x2, y: y2 };

if (isNaN(y2)) {
    console.log(-1)
    return -1
}
function computeCameraParameters(imageWidth, imageHeight, vp1, vp2, principalPoint) {
    vp1 = coordinates_util_1.default.convert(vp1, coordinates_util_1.ImageCoordinateFrame.Absolute, coordinates_util_1.ImageCoordinateFrame.ImagePlane, imageWidth, imageHeight);
    vp2 = coordinates_util_1.default.convert(vp2, coordinates_util_1.ImageCoordinateFrame.Absolute, coordinates_util_1.ImageCoordinateFrame.ImagePlane, imageWidth, imageHeight);
    var controlPointsStateBase = {
        principalPoint: principalPoint,
        origin: { x: 0.5, y: 0.8 },
        referenceDistanceAnchor: { x: 0.5, y: 0.8 },
        firstVanishingPoint: { lineSegments: [[{ x: 0, y: 0 }, { x: 0, y: 0 }], [{ x: 1, y: 1 }, { x: 0, y: 0 }]] },
        referenceDistanceHandleOffsets: [0.131, 0.605],
    };
    var relativeFocalLength = solve_1.default.computeFocalLength(vp1, vp2, principalPoint);
    // @ts-ignore
    return solve_1.default.computeCameraParameters(controlPointsStateBase,
                                                   calibration_settings_1.defaultCalibrationSettingsBase,
                                                   principalPoint,
                                                   vp1, vp2,
                                                   relativeFocalLength,
                                                   imageWidth, imageHeight);
}
console.log(JSON.stringify(computeCameraParameters(imageWidth, imageHeight, vp1, vp2, principalPoint)))
return -1
