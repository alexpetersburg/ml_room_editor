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
exports.ReferenceDistanceUnit = exports.Axis = exports.PrincipalPointMode2VP = exports.PrincipalPointMode1VP = void 0;
var PrincipalPointMode1VP;
(function (PrincipalPointMode1VP) {
    PrincipalPointMode1VP["Default"] = "Default";
    PrincipalPointMode1VP["Manual"] = "Manual";
})(PrincipalPointMode1VP = exports.PrincipalPointMode1VP || (exports.PrincipalPointMode1VP = {}));
var PrincipalPointMode2VP;
(function (PrincipalPointMode2VP) {
    PrincipalPointMode2VP["Default"] = "Default";
    PrincipalPointMode2VP["Manual"] = "Manual";
    PrincipalPointMode2VP["FromThirdVanishingPoint"] = "FromThirdVanishingPoint";
})(PrincipalPointMode2VP = exports.PrincipalPointMode2VP || (exports.PrincipalPointMode2VP = {}));
var Axis;
(function (Axis) {
    Axis["PositiveX"] = "xPositive";
    Axis["NegativeX"] = "xNegative";
    Axis["PositiveY"] = "yPositive";
    Axis["NegativeY"] = "yNegative";
    Axis["PositiveZ"] = "zPositive";
    Axis["NegativeZ"] = "zNegative";
})(Axis = exports.Axis || (exports.Axis = {}));
var ReferenceDistanceUnit;
(function (ReferenceDistanceUnit) {
    ReferenceDistanceUnit["None"] = "No unit";
    ReferenceDistanceUnit["Millimeters"] = "Millimeters";
    ReferenceDistanceUnit["Centimeters"] = "Centimeters";
    ReferenceDistanceUnit["Meters"] = "Meters";
    ReferenceDistanceUnit["Kilometers"] = "Kilometers";
    ReferenceDistanceUnit["Inches"] = "Inches";
    ReferenceDistanceUnit["Feet"] = "Feet";
    ReferenceDistanceUnit["Miles"] = "Miles";
})(ReferenceDistanceUnit = exports.ReferenceDistanceUnit || (exports.ReferenceDistanceUnit = {}));
