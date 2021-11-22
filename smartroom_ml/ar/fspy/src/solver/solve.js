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
var __assign = (this && this.__assign) || function () {
    __assign = Object.assign || function(t) {
        for (var s, i = 1, n = arguments.length; i < n; i++) {
            s = arguments[i];
            for (var p in s) if (Object.prototype.hasOwnProperty.call(s, p))
                t[p] = s[p];
        }
        return t;
    };
    return __assign.apply(this, arguments);
};
Object.defineProperty(exports, "__esModule", { value: true });
var calibration_settings_1 = require("../types/calibration-settings");
var math_util_1 = require("./math-util");
var transform_1 = require("./transform");
var vector_3d_1 = require("./vector-3d");
var coordinates_util_1 = require("./coordinates-util");
var solver_result_1 = require("../defaults/solver-result");
var camera_presets_1 = require("./camera-presets");
var strings_1 = require("../strings/strings");
/**
 * The solver handles estimation of focal length and camera orientation
 * from given vanishing points. Sections numbers, equations numbers etc
 * refer to "Using Vanishing Points for Camera Calibration and Coarse 3D Reconstruction
 * from a Single Image" by E. Guillou, D. Meneveaux, E. Maisel, K. Bouatouch.
 * @see http://www.irisa.fr/prive/kadi/Reconstruction/paper.ps.gz
 */
var Solver = /** @class */ (function () {
    function Solver() {
    }
    /**
     * Estimates camera parameters given a single vanishing point, a
     * relative focal length and an optional horizon direction
     * @param settings
     * @param controlPoints
     * @param image
     */
    Solver.solve1VP = function (settingsBase, settings1VP, controlPointsBase, controlPoints1VP, image) {
        // Create a blank result object
        var result = this.blankSolverResult();
        // Bail out if we don't have valid image dimensions
        result.errors = this.validateImageDimensions(image);
        if (result.errors.length > 0) {
            return result;
        }
        var imageWidth = image.width;
        var imageHeight = image.height;
        // Compute a relative focal length from the provided absolute focal length and sensor size
        var absoluteFocalLength = settings1VP.absoluteFocalLength;
        var sensorWidth = settingsBase.cameraData.customSensorWidth;
        var sensorHeight = settingsBase.cameraData.customSensorHeight;
        var presetId = settingsBase.cameraData.presetId;
        if (presetId) {
            var preset = camera_presets_1.cameraPresets[presetId];
            sensorWidth = preset.sensorWidth;
            sensorHeight = preset.sensorHeight;
        }
        var relativeFocalLength = 0;
        var sensorAspectRatio = sensorWidth / sensorHeight;
        // TODO: verify factor 2
        if (sensorAspectRatio > 1) {
            // wide sensor.
            relativeFocalLength = 2 * absoluteFocalLength / sensorWidth;
        }
        else {
            // tall sensor
            relativeFocalLength = 2 * absoluteFocalLength / sensorHeight;
        }
        if (!this.imageProportionsMatchSensor(sensorWidth, sensorHeight, imageWidth, imageHeight)) {
            result.warnings.push(strings_1.default.imageSensorProportionsMismatch);
        }
        // Compute the input vanishing point in image plane coordinates
        var vanishingPointStates = [controlPointsBase.firstVanishingPoint];
        var inputVanishingPoints = this.computeVanishingPointsFromControlPoints(image, vanishingPointStates, result.errors);
        this.validateVanishingPointAccuracy(vanishingPointStates, result.warnings);
        if (result.errors.length > 0) {
            // Something went wrong computing the vanishing point. Nothing further.
            return result;
        }
        // Get the principal point
        var principalPoint = { x: 0, y: 0 };
        if (settings1VP.principalPointMode == calibration_settings_1.PrincipalPointMode1VP.Manual) {
            principalPoint = coordinates_util_1.default.convert(controlPointsBase.principalPoint, coordinates_util_1.ImageCoordinateFrame.Relative, coordinates_util_1.ImageCoordinateFrame.ImagePlane, imageWidth, imageHeight);
        }
        // Compute the horizon direction
        var horizonDirection = { x: 1, y: 0 }; // flat by default
        // Compute two points on the horizon line in image plane coordinates
        var horizonStart = coordinates_util_1.default.convert(controlPoints1VP.horizon[0], coordinates_util_1.ImageCoordinateFrame.Relative, coordinates_util_1.ImageCoordinateFrame.ImagePlane, imageWidth, imageHeight);
        var horizonEnd = coordinates_util_1.default.convert(controlPoints1VP.horizon[1], coordinates_util_1.ImageCoordinateFrame.Relative, coordinates_util_1.ImageCoordinateFrame.ImagePlane, imageWidth, imageHeight);
        // Normalized horizon direction vector
        horizonDirection = math_util_1.default.normalized({
            x: horizonEnd.x - horizonStart.x,
            y: horizonEnd.y - horizonStart.y
        });
        var secondVanishingPoint = this.computeSecondVanishingPoint(inputVanishingPoints[0], relativeFocalLength, principalPoint, horizonDirection);
        if (secondVanishingPoint === null) {
            result.errors.push('Failed to compute second vanishing point');
            return result;
        }
        result.cameraParameters = this.computeCameraParameters(controlPointsBase, settingsBase, principalPoint, inputVanishingPoints[0], secondVanishingPoint, relativeFocalLength, imageWidth, imageHeight);
        return result;
    };
    Solver.solve2VP = function (settingsBase, settings2VP, controlPointsBase, controlPoints2VP, image) {
        var result = this.blankSolverResult();
        var errors = this.validateImageDimensions(image);
        if (errors.length > 0) {
            result.errors = errors;
            return result;
        }
        var imageWidth = image.width;
        var imageHeight = image.height;
        var firstVanishingPointControlState = controlPointsBase.firstVanishingPoint;
        var secondVanishingPointControlState = controlPoints2VP.secondVanishingPoint;
        if (settings2VP.quadModeEnabled) {
            secondVanishingPointControlState = {
                lineSegments: [
                    [
                        firstVanishingPointControlState.lineSegments[0][0],
                        firstVanishingPointControlState.lineSegments[1][0]
                    ],
                    [
                        firstVanishingPointControlState.lineSegments[0][1],
                        firstVanishingPointControlState.lineSegments[1][1]
                    ]
                ]
            };
        }
        // Compute the two input vanishing points from the provided control points
        var inputVanishingPoints = this.computeVanishingPointsFromControlPoints(image, [controlPointsBase.firstVanishingPoint, secondVanishingPointControlState], errors);
        if (!inputVanishingPoints) {
            result.errors = errors;
            return result;
        }
        // Get the principal point
        var principalPoint = { x: 0, y: 0 };
        switch (settings2VP.principalPointMode) {
            case calibration_settings_1.PrincipalPointMode2VP.Manual:
                principalPoint = coordinates_util_1.default.convert(controlPointsBase.principalPoint, coordinates_util_1.ImageCoordinateFrame.Relative, coordinates_util_1.ImageCoordinateFrame.ImagePlane, image.width, image.height);
                break;
            case calibration_settings_1.PrincipalPointMode2VP.FromThirdVanishingPoint:
                var thirdVanishingPointArray = this.computeVanishingPointsFromControlPoints(image, [controlPoints2VP.thirdVanishingPoint], errors);
                if (thirdVanishingPointArray) {
                    var thirdVanishingPoint = thirdVanishingPointArray[0];
                    principalPoint = math_util_1.default.triangleOrthoCenter(inputVanishingPoints[0], inputVanishingPoints[1], thirdVanishingPoint);
                }
                break;
        }
        var fRelative = this.computeFocalLength(inputVanishingPoints[0], inputVanishingPoints[1], principalPoint);
        if (fRelative === null) {
            result.errors.push('Invalid vanishing point configuration. Failed to compute focal length.');
            return result;
        }
        // Check vanishing point accuracy
        var vanishingPointStatesToCheck = [controlPointsBase.firstVanishingPoint, secondVanishingPointControlState];
        if (settings2VP.principalPointMode == calibration_settings_1.PrincipalPointMode2VP.FromThirdVanishingPoint) {
            vanishingPointStatesToCheck.push(controlPoints2VP.thirdVanishingPoint);
        }
        this.validateVanishingPointAccuracy(vanishingPointStatesToCheck, result.warnings);
        // compute camera parameters
        result.cameraParameters = this.computeCameraParameters(controlPointsBase, settingsBase, principalPoint, inputVanishingPoints[0], inputVanishingPoints[1], fRelative, imageWidth, imageHeight);
        return result;
    };
    Solver.imageProportionsMatchSensor = function (imageWidth, imageHeight, sensorWidth, sensorHeight) {
        if (sensorHeight == 0 || sensorWidth == 0 || imageWidth == 0 || imageHeight == 0) {
            return false;
        }
        var epsilon = 0.01;
        var imageAspectRatio = imageWidth / imageHeight;
        var sensorAspectRatio = sensorWidth / sensorHeight;
        var imageFitsSensor = Math.abs(sensorAspectRatio - imageAspectRatio) < epsilon;
        var rotated90DegImageFitsSensor = Math.abs(sensorAspectRatio - 1 / imageAspectRatio) < epsilon;
        if (!imageFitsSensor && !rotated90DegImageFitsSensor) {
            return false;
        }
        return true;
    };
    /**
     * Computes the focal length based on two vanishing points and a center of projection.
     * See 3.2 "Determining the focal length from a single image"
     * @param Fu the first vanishing point in image plane coordinates.
     * @param Fv the second vanishing point in image plane coordinates.
     * @param P the center of projection in image plane coordinates.
     * @returns The relative focal length.
     */
    Solver.computeFocalLength = function (Fu, Fv, P) {
        // compute Puv, the orthogonal projection of P onto FuFv
        var dirFuFv = new vector_3d_1.default(Fu.x - Fv.x, Fu.y - Fv.y).normalized();
        var FvP = new vector_3d_1.default(P.x - Fv.x, P.y - Fv.y);
        var proj = dirFuFv.dot(FvP);
        var Puv = {
            x: proj * dirFuFv.x + Fv.x,
            y: proj * dirFuFv.y + Fv.y
        };
        var PPuv = new vector_3d_1.default(P.x - Puv.x, P.y - Puv.y).length;
        var FvPuv = new vector_3d_1.default(Fv.x - Puv.x, Fv.y - Puv.y).length;
        var FuPuv = new vector_3d_1.default(Fu.x - Puv.x, Fu.y - Puv.y).length;
        var fSq = FvPuv * FuPuv - PPuv * PPuv;
        if (fSq <= 0) {
            return null;
        }
        return Math.sqrt(fSq);
    };
    /**
     * Computes the camera rotation matrix based on two vanishing points
     * and a focal length as in section 3.3 "Computing the rotation matrix".
     * @param Fu the first vanishing point in normalized image coordinates.
     * @param Fv the second vanishing point in normalized image coordinates.
     * @param f the relative focal length.
     * @param P the principal point
     * @returns The matrix Moc
     */
    Solver.computeCameraRotationMatrix = function (Fu, Fv, f, P) {
        var OFu = new vector_3d_1.default(Fu.x - P.x, Fu.y - P.y, -f);
        var OFv = new vector_3d_1.default(Fv.x - P.x, Fv.y - P.y, -f);
        var s1 = OFu.length;
        var upRc = OFu.normalized();
        var s2 = OFv.length;
        var vpRc = OFv.normalized();
        var wpRc = upRc.cross(vpRc);
        var M = new transform_1.default();
        M.matrix[0][0] = OFu.x / s1;
        M.matrix[0][1] = OFv.x / s2;
        M.matrix[0][2] = wpRc.x;
        M.matrix[1][0] = OFu.y / s1;
        M.matrix[1][1] = OFv.y / s2;
        M.matrix[1][2] = wpRc.y;
        M.matrix[2][0] = -f / s1;
        M.matrix[2][1] = -f / s2;
        M.matrix[2][2] = wpRc.z;
        return M;
    };
    Solver.vanishingPointIndexForAxis = function (positiveAxis, vanishingPointAxes) {
        var negativeAxis = calibration_settings_1.Axis.NegativeX;
        switch (positiveAxis) {
            case calibration_settings_1.Axis.PositiveY:
                negativeAxis = calibration_settings_1.Axis.NegativeY;
                break;
            case calibration_settings_1.Axis.PositiveZ:
                negativeAxis = calibration_settings_1.Axis.NegativeZ;
                break;
        }
        for (var vpIndex = 0; vpIndex < 3; vpIndex++) {
            var vpAxis = vanishingPointAxes[vpIndex];
            if (vpAxis == positiveAxis || vpAxis == negativeAxis) {
                return vpIndex;
            }
        }
        return 0;
    };
    Solver.referenceDistanceHandlesWorldPositions = function (controlPoints, referenceAxis, imageWidth, imageHeight, cameraParameters) {
        var handlePositionsRelative = this.referenceDistanceHandlesRelativePositions(controlPoints, referenceAxis, cameraParameters.vanishingPoints, cameraParameters.vanishingPointAxes, imageWidth, imageHeight);
        // handle positions in image plane coordinates
        var handlePositions = [
            coordinates_util_1.default.convert(handlePositionsRelative[0], coordinates_util_1.ImageCoordinateFrame.Relative, coordinates_util_1.ImageCoordinateFrame.ImagePlane, imageWidth, imageHeight),
            coordinates_util_1.default.convert(handlePositionsRelative[1], coordinates_util_1.ImageCoordinateFrame.Relative, coordinates_util_1.ImageCoordinateFrame.ImagePlane, imageWidth, imageHeight)
        ];
        // anchor position in image plane coordinates
        var anchorPosition = coordinates_util_1.default.convert(controlPoints.referenceDistanceAnchor, coordinates_util_1.ImageCoordinateFrame.Relative, coordinates_util_1.ImageCoordinateFrame.ImagePlane, imageWidth, imageHeight);
        // Two vectors u, v spanning the reference plane, i.e the plane
        // perpendicular to the reference axis w
        var origin = new vector_3d_1.default();
        var u = new vector_3d_1.default();
        var v = new vector_3d_1.default();
        var w = new vector_3d_1.default();
        switch (referenceAxis) {
            case calibration_settings_1.Axis.PositiveX:
                u.y = 1;
                v.z = 1;
                w.x = 1;
                break;
            case calibration_settings_1.Axis.PositiveY:
                u.x = 1;
                v.z = 1;
                w.y = 1;
                break;
            case calibration_settings_1.Axis.PositiveZ:
                u.x = 1;
                v.y = 1;
                w.z = 1;
                break;
        }
        // The reference distance anchor is defined to lie in the reference plane p.
        // Let rayAnchor be a ray from the camera through the anchor position in the image plane.
        // The intersection of p and rayAnchor give us two coordinate values u0 and v0.
        var rayAnchorStart = math_util_1.default.perspectiveUnproject(new vector_3d_1.default(anchorPosition.x, anchorPosition.y, 1), cameraParameters.viewTransform, cameraParameters.principalPoint, cameraParameters.horizontalFieldOfView);
        var rayAnchorEnd = math_util_1.default.perspectiveUnproject(new vector_3d_1.default(anchorPosition.x, anchorPosition.y, 2), cameraParameters.viewTransform, cameraParameters.principalPoint, cameraParameters.horizontalFieldOfView);
        var referencePlaneIntersection = math_util_1.default.linePlaneIntersection(origin, u, v, rayAnchorStart, rayAnchorEnd);
        // Compute the world positions of the reference distance handles
        var result = [];
        for (var _i = 0, handlePositions_1 = handlePositions; _i < handlePositions_1.length; _i++) {
            var handlePosition = handlePositions_1[_i];
            var handleRayStart = math_util_1.default.perspectiveUnproject(new vector_3d_1.default(handlePosition.x, handlePosition.y, 1), cameraParameters.viewTransform, cameraParameters.principalPoint, cameraParameters.horizontalFieldOfView);
            var handleRayEnd = math_util_1.default.perspectiveUnproject(new vector_3d_1.default(handlePosition.x, handlePosition.y, 2), cameraParameters.viewTransform, cameraParameters.principalPoint, cameraParameters.horizontalFieldOfView);
            var handlePosition3D = math_util_1.default.shortestLineSegmentBetweenLines(handleRayStart, handleRayEnd, referencePlaneIntersection, referencePlaneIntersection.added(w))[0];
            result.push(handlePosition3D);
        }
        return [result[0], result[1]];
    };
    Solver.referenceDistanceHandlesRelativePositions = function (controlPoints, referenceAxis, vanishingPoints, vanishingPointAxes, imageWidth, imageHeight) {
        // The position of the reference distance anchor in relative coordinates
        var anchor = controlPoints.referenceDistanceAnchor;
        // The index of the vanishing point corresponding to the reference axis
        var vpIndex = this.vanishingPointIndexForAxis(referenceAxis, vanishingPointAxes);
        // The position of the vanishing point in relative coordinates
        var vp = coordinates_util_1.default.convert(vanishingPoints[vpIndex], coordinates_util_1.ImageCoordinateFrame.ImagePlane, coordinates_util_1.ImageCoordinateFrame.Relative, imageWidth, imageHeight);
        // A unit vector pointing from the anchor to the vanishing point
        var anchorToVp = math_util_1.default.normalized({ x: vp.x - anchor.x, y: vp.y - anchor.y });
        // The handles lie on the line from the anchor to the vanishing point
        var handleOffsets = controlPoints.referenceDistanceHandleOffsets;
        return [
            {
                x: anchor.x + handleOffsets[0] * anchorToVp.x,
                y: anchor.y + handleOffsets[0] * anchorToVp.y
            },
            {
                x: anchor.x + handleOffsets[1] * anchorToVp.x,
                y: anchor.y + handleOffsets[1] * anchorToVp.y
            }
        ];
    };
    /**
     * Computes the coordinates of the second vanishing point
     * based on the first, a focal length, the center of projection and
     * the desired horizon tilt angle. The equations here are derived from
     * section 3.2 "Determining the focal length from a single image".
     * @param Fu the first vanishing point in image plane coordinates.
     * @param f the relative focal length
     * @param P the center of projection in normalized image coordinates
     * @param horizonDir The desired horizon direction
     */
    Solver.computeSecondVanishingPoint = function (Fu, f, P, horizonDir) {
        // find the second vanishing point
        // TODO_ take principal point into account here
        if (math_util_1.default.distance(Fu, P) < 1e-7) { // TODO: espsilon constant
            return null;
        }
        var Fup = {
            x: Fu.x - P.x,
            y: Fu.y - P.y
        };
        var k = -(Fup.x * Fup.x + Fup.y * Fup.y + f * f) / (Fup.x * horizonDir.x + Fup.y * horizonDir.y);
        var Fv = {
            x: Fup.x + k * horizonDir.x + P.x,
            y: Fup.y + k * horizonDir.y + P.y
        };
        return Fv;
    };
    Solver.axisVector = function (axis) {
        switch (axis) {
            case calibration_settings_1.Axis.NegativeX:
                return new vector_3d_1.default(-1, 0, 0);
            case calibration_settings_1.Axis.PositiveX:
                return new vector_3d_1.default(1, 0, 0);
            case calibration_settings_1.Axis.NegativeY:
                return new vector_3d_1.default(0, -1, 0);
            case calibration_settings_1.Axis.PositiveY:
                return new vector_3d_1.default(0, 1, 0);
            case calibration_settings_1.Axis.NegativeZ:
                return new vector_3d_1.default(0, 0, -1);
            case calibration_settings_1.Axis.PositiveZ:
                return new vector_3d_1.default(0, 0, 1);
        }
    };
    Solver.vectorAxis = function (vector) {
        if (vector.x == 0 && vector.y == 0) {
            return vector.z > 0 ? calibration_settings_1.Axis.PositiveZ : calibration_settings_1.Axis.NegativeZ;
        }
        else if (vector.x == 0 && vector.z == 0) {
            return vector.y > 0 ? calibration_settings_1.Axis.PositiveY : calibration_settings_1.Axis.NegativeY;
        }
        else if (vector.y == 0 && vector.z == 0) {
            return vector.x > 0 ? calibration_settings_1.Axis.PositiveX : calibration_settings_1.Axis.NegativeX;
        }
        throw new Error('Invalid axis vector');
    };
    Solver.validateImageDimensions = function (image) {
        var errors = [];
        if (image.width === null || image.height === null) {
            errors.push('No image loaded');
        }
        return errors;
    };
    Solver.validateVanishingPointAccuracy = function (controlPointStates, warnings) {
        controlPointStates.forEach(function (controlPointState, stateIndex) {
            var line1Direction = math_util_1.default.normalized(math_util_1.default.difference(controlPointState.lineSegments[0][1], controlPointState.lineSegments[0][0]));
            var line2Direction = math_util_1.default.normalized(math_util_1.default.difference(controlPointState.lineSegments[1][1], controlPointState.lineSegments[1][0]));
            var dot = math_util_1.default.dot(line1Direction, line2Direction);
            if (Math.abs(dot) > 0.99995) {
                warnings.push('Near parallel lines for VP ' + (stateIndex + 1));
            }
        });
    };
    /**
     * Computes vanishing points in image plane coordinates given a set of
     * vanishing point control points.
     * @param image
     * @param controlPointStates
     * @param errors
     */
    Solver.computeVanishingPointsFromControlPoints = function (image, controlPointStates, errors) {
        var result = [];
        for (var i = 0; i < controlPointStates.length; i++) {
            var vanishingPoint = math_util_1.default.lineIntersection(controlPointStates[i].lineSegments[0], controlPointStates[i].lineSegments[1]);
            if (vanishingPoint) {
                result.push(coordinates_util_1.default.convert(vanishingPoint, coordinates_util_1.ImageCoordinateFrame.Relative, coordinates_util_1.ImageCoordinateFrame.ImagePlane, image.width, image.height));
            }
            else {
                errors.push('Failed to compute vanishing point');
            }
        }
        return errors.length == 0 ? result : null;
    };
    Solver.computeTranslationVector = function (controlPoints, settings, imageWidth, imageHeight, cameraParameters) {
        // The 3D origin in image plane coordinates
        var origin = coordinates_util_1.default.convert(controlPoints.origin, coordinates_util_1.ImageCoordinateFrame.Relative, coordinates_util_1.ImageCoordinateFrame.ImagePlane, imageWidth, imageHeight);
        var k = Math.tan(0.5 * cameraParameters.horizontalFieldOfView);
        var origin3D = new vector_3d_1.default(k * (origin.x - cameraParameters.principalPoint.x), k * (origin.y - cameraParameters.principalPoint.y), -1).multipliedByScalar(this.DEFAULT_CAMERA_DISTANCE_SCALE);
        // Set a default translation vector
        cameraParameters.viewTransform.matrix[0][3] = origin3D.x;
        cameraParameters.viewTransform.matrix[1][3] = origin3D.y;
        cameraParameters.viewTransform.matrix[2][3] = origin3D.z;
        if (settings.referenceDistanceAxis) {
            // If requested, scale the translation vector so that
            // the distance between the 3d handle positions equals the
            // specified reference distance
            // See what the distance between the 3d handle positions is given the current,
            // default, translation vector
            var referenceDistanceHandles3D = this.referenceDistanceHandlesWorldPositions(controlPoints, settings.referenceDistanceAxis, imageWidth, imageHeight, cameraParameters);
            var defaultHandleDistance = referenceDistanceHandles3D[0].subtracted(referenceDistanceHandles3D[1]).length;
            // Scale the translation vector by the ratio of the reference distance to the computed distance
            var referenceDistance = settings.referenceDistance;
            var scale = referenceDistance / defaultHandleDistance;
            origin3D.multiplyByScalar(scale);
        }
        cameraParameters.viewTransform.matrix[0][3] = origin3D.x;
        cameraParameters.viewTransform.matrix[1][3] = origin3D.y;
        cameraParameters.viewTransform.matrix[2][3] = origin3D.z;
    };
    /**
     * Creates a blank solver result to be populated by the solver
     */
    Solver.blankSolverResult = function () {
        var result = __assign({}, solver_result_1.defaultSolverResult);
        result.errors = [];
        result.warnings = [];
        return result;
    };
    Solver.computeFieldOfView = function (imageWidth, imageHeight, fRelative, vertical) {
        var aspectRatio = imageWidth / imageHeight;
        var d = vertical ? 1 / aspectRatio : 1;
        return 2 * Math.atan(d / fRelative);
    };
    Solver.computeCameraParameters = function (controlPoints, settings, principalPoint, vp1, vp2, relativeFocalLength, imageWidth, imageHeight) {
        var cameraParameters = {
            principalPoint: { x: 0, y: 0 },
            viewTransform: new transform_1.default(),
            cameraTransform: new transform_1.default(),
            horizontalFieldOfView: 0,
            verticalFieldOfView: 0,
            vanishingPoints: [{ x: 0, y: 0 }, { x: 0, y: 0 }, { x: 0, y: 0 }],
            vanishingPointAxes: [calibration_settings_1.Axis.NegativeX, calibration_settings_1.Axis.NegativeX, calibration_settings_1.Axis.NegativeX],
            relativeFocalLength: 0,
            imageWidth: imageWidth,
            imageHeight: imageHeight
        };
        // Assing vanishing point axes
        var axisAssignmentMatrix = new transform_1.default();
        var row1 = this.axisVector(settings.firstVanishingPointAxis);
        var row2 = this.axisVector(settings.secondVanishingPointAxis);
        var row3 = row1.cross(row2);
        axisAssignmentMatrix.matrix[0][0] = row1.x;
        axisAssignmentMatrix.matrix[0][1] = row1.y;
        axisAssignmentMatrix.matrix[0][2] = row1.z;
        axisAssignmentMatrix.matrix[1][0] = row2.x;
        axisAssignmentMatrix.matrix[1][1] = row2.y;
        axisAssignmentMatrix.matrix[1][2] = row2.z;
        axisAssignmentMatrix.matrix[2][0] = row3.x;
        axisAssignmentMatrix.matrix[2][1] = row3.y;
        axisAssignmentMatrix.matrix[2][2] = row3.z;
        if (Math.abs(1 - axisAssignmentMatrix.determinant) > 1e-7) { // TODO: eps
            console.log(1);
            return null;
        }
        cameraParameters.vanishingPointAxes = [
            settings.firstVanishingPointAxis,
            settings.secondVanishingPointAxis,
            this.vectorAxis(row3)
        ];
        // principal point
        cameraParameters.principalPoint = principalPoint;
        // focal length
        cameraParameters.relativeFocalLength = relativeFocalLength;
        // vanishing points
        cameraParameters.vanishingPoints = [
            vp1,
            vp2,
            math_util_1.default.thirdTriangleVertex(vp1, vp2, principalPoint)
        ];
        // horizontal field of view
        cameraParameters.horizontalFieldOfView = this.computeFieldOfView(imageWidth, imageHeight, relativeFocalLength, false);
        // vertical field of view
        cameraParameters.verticalFieldOfView = this.computeFieldOfView(imageWidth, imageHeight, relativeFocalLength, true);
        // compute camera rotation matrix
        var cameraRotationMatrix = this.computeCameraRotationMatrix(vp1, vp2, relativeFocalLength, principalPoint);
        if (Math.abs(cameraRotationMatrix.determinant - 1) > 1e-2) { // TODO: eps
            console.log(cameraRotationMatrix.determinant.toFixed(5));
            console.log('wrong cameraRotationMatrix.determinant');
            return null;
        }
        cameraParameters.viewTransform = axisAssignmentMatrix.leftMultiplied(cameraRotationMatrix);
        this.computeTranslationVector(controlPoints, settings, imageWidth, imageHeight, cameraParameters);
        cameraParameters.cameraTransform = cameraParameters.viewTransform.inverted();
        return cameraParameters;
    };
    Solver.DEFAULT_CAMERA_DISTANCE_SCALE = 10;
    return Solver;
}());
exports.default = Solver;
