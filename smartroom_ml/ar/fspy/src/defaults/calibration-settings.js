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
exports.defaultCalibrationSettings2VP = exports.defaultCalibrationSettings1VP = exports.defaultCalibrationSettingsBase = void 0;
var calibration_settings_1 = require("../types/calibration-settings");
exports.defaultCalibrationSettingsBase = {
    referenceDistanceAxis: null,
    referenceDistance: 4,
    referenceDistanceUnit: calibration_settings_1.ReferenceDistanceUnit.Meters,
    cameraData: {
        presetId: null,
        customSensorWidth: 36,
        customSensorHeight: 24
    },
    firstVanishingPointAxis: calibration_settings_1.Axis.PositiveX,
    secondVanishingPointAxis: calibration_settings_1.Axis.PositiveY
};
exports.defaultCalibrationSettings1VP = {
    principalPointMode: calibration_settings_1.PrincipalPointMode1VP.Default,
    absoluteFocalLength: 24
};
exports.defaultCalibrationSettings2VP = {
    principalPointMode: calibration_settings_1.PrincipalPointMode2VP.Default,
    quadModeEnabled: false
};
