import Solver from './src/solver/solve'
import MathUtil from "./src/solver/math-util";
import {defaultCalibrationSettingsBase} from "./src/defaults/calibration-settings";
import CoordinatesUtil, {ImageCoordinateFrame} from "./src/solver/coordinates-util";
import {CameraParameters} from "./src/solver/solver-result";
import Point2D from "./src/solver/point-2d";

let imageWidth = 1170
let imageHeight = 780
let vp1 = {x:-118.47391956827687, y: 384.3497574239535 }
let vp2 = {x:1080.5994814938165     , y: 397.6670855398042 }
let vp3 = {x:-2880.818096359069     , y: 102.27408808118403 }
let principalPoint = MathUtil.triangleOrthoCenter(
    CoordinatesUtil.convert(vp1, ImageCoordinateFrame.Absolute, ImageCoordinateFrame.Relative, imageWidth, imageHeight),
    CoordinatesUtil.convert(vp2, ImageCoordinateFrame.Absolute, ImageCoordinateFrame.Relative, imageWidth, imageHeight),
    CoordinatesUtil.convert(vp3, ImageCoordinateFrame.Absolute, ImageCoordinateFrame.Relative, imageWidth, imageHeight)
)
principalPoint = {x: 0.5, y:0.5}


function computeCameraParameters(
    imageWidth: number,
    imageHeight: number,
    vp1: Point2D,
    vp2: Point2D,
    principalPoint
): CameraParameters | null {
    vp1 = CoordinatesUtil.convert(vp1, ImageCoordinateFrame.Absolute, ImageCoordinateFrame.Relative, imageWidth, imageHeight)
    vp2 = CoordinatesUtil.convert(vp2, ImageCoordinateFrame.Absolute, ImageCoordinateFrame.Relative, imageWidth, imageHeight)
    let controlPointsStateBase = {
        principalPoint: principalPoint,
        origin: {x: 0.5,y:0.8},
        referenceDistanceAnchor: {x: 0.5,y: 0.8},
        firstVanishingPoint: {lineSegments: [[{x: 0, y: 0},{x: 0, y: 0}],[{x: 0, y: 0},{x: 0, y: 0}]]},
        referenceDistanceHandleOffsets: [-0.235, -0.306],
    }
    let relativeFocalLength = Solver.computeFocalLength(vp1, vp2, principalPoint)
    // @ts-ignore
    return Solver.computeCameraParameters(controlPointsStateBase, defaultCalibrationSettingsBase, principalPoint, vp1, vp2 ,relativeFocalLength,
        imageWidth, imageHeight)
}
console.log(JSON.stringify(computeCameraParameters(imageWidth, imageHeight, vp1, vp2, principalPoint)))