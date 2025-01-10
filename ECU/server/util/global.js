/**
 * @file global.js
 * @description Defines global variables, states, and configurations used across the application.
 * @author Lior Jigalo
 * @license MIT
 */

/** Import path module */
const path = require("path");

/**
 * Locks used for controlling various application states.
 * @type {{alert_lock: boolean}}
 */
let locks = {
    alert_lock: false
}

/**
 * Represents the current state of the car.
 * @type {{canceled: boolean,
 *         stopped: boolean,
 *         decelerating: boolean,
 *         accelerating: boolean,
 *         cruising: boolean
 *        }}
 */
let carState = {
    accelerating: true,
    decelerating: false,
    cruising: false,
    stopped: false,
    canceled: false
}

/**
 * Paths to sound assets used for various alerts and notifications.
 * @type {{decelerateWarning: string,
 *         failedToParse: string,
 *         decelerating: string,
 *         gotIt: string,
 *         attentionTest: string,
 *         noResponse: string,
 *         takeABreak: string
 *        }}
 */
const sounds = {
    takeABreak:         path.join(__dirname, '../assets/sounds', "takeABreak.wav"),
    attentionTest:      path.join(__dirname, '../assets/sounds', 'attentionTest.wav'),
    failedToParse:      path.join(__dirname, '../assets/sounds', 'failedToParse.wav'),
    noResponse:         path.join(__dirname, '../assets/sounds', 'noResponse.wav'),
    decelerateWarning:  path.join(__dirname, '../assets/sounds', 'decelerateWarning.wav'),
    decelerating:       path.join(__dirname, '../assets/sounds', 'decelerating.wav'),
    gotIt:              path.join(__dirname, '../assets/sounds', 'gotIt.wav'),
};

/**
 * User response statuses for various prompts or actions.
 * @type {{ userResponded: number,
 *          failedToParse: number,
 *          noResponse: number
 *        }}
 */
const user_status = {
    noResponse: 0,
    userResponded: 1,
    failedToParse: 2
}

/** Export global variables and configurations */
module.exports = {locks, carState, user_status, sounds}