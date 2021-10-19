"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
var solve_1 = require("./src/solver/solve");
var math_util_1 = require("./src/solver/math-util");
var calibration_settings_1 = require("./src/defaults/calibration-settings");
var coordinates_util_1 = require("./src/solver/coordinates-util");
var imageWidth = 1170;
var imageHeight = 780;
var vp1 = { x: -118.47391956827687, y: 384.3497574239535 };
var vp2 = { x: 1080.5994814938165, y: 397.6670855398042 };
var vp3 = { x: -2880.818096359069, y: 102.27408808118403 };
var principalPoint = math_util_1.default.triangleOrthoCenter(coordinates_util_1.default.convert(vp1, coordinates_util_1.ImageCoordinateFrame.Absolute, coordinates_util_1.ImageCoordinateFrame.Relative, imageWidth, imageHeight), coordinates_util_1.default.convert(vp2, coordinates_util_1.ImageCoordinateFrame.Absolute, coordinates_util_1.ImageCoordinateFrame.Relative, imageWidth, imageHeight), coordinates_util_1.default.convert(vp3, coordinates_util_1.ImageCoordinateFrame.Absolute, coordinates_util_1.ImageCoordinateFrame.Relative, imageWidth, imageHeight));
principalPoint = { x: 0.5, y: 0.5 };
function computeCameraParameters(imageWidth, imageHeight, vp1, vp2, principalPoint) {
    vp1 = coordinates_util_1.default.convert(vp1, coordinates_util_1.ImageCoordinateFrame.Absolute, coordinates_util_1.ImageCoordinateFrame.Relative, imageWidth, imageHeight);
    vp2 = coordinates_util_1.default.convert(vp2, coordinates_util_1.ImageCoordinateFrame.Absolute, coordinates_util_1.ImageCoordinateFrame.Relative, imageWidth, imageHeight);
    var controlPointsStateBase = {
        principalPoint: principalPoint,
        origin: { x: 0.5, y: 0.8 },
        referenceDistanceAnchor: { x: 0.5, y: 0.8 },
        firstVanishingPoint: { lineSegments: [[{ x: 0, y: 0 }, { x: 0, y: 0 }], [{ x: 0, y: 0 }, { x: 0, y: 0 }]] },
        referenceDistanceHandleOffsets: [-0.235, -0.306],
    };
    var relativeFocalLength = solve_1.default.computeFocalLength(vp1, vp2, principalPoint);
    // @ts-ignore
    return solve_1.default.computeCameraParameters(controlPointsStateBase, calibration_settings_1.defaultCalibrationSettingsBase, principalPoint, vp1, vp2, relativeFocalLength, imageWidth, imageHeight);
}
console.log(JSON.stringify(computeCameraParameters(imageWidth, imageHeight, vp1, vp2, principalPoint)));
