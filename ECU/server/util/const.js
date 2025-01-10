/**
 * @file constants.js
 * @description Contains application-wide constants and predefined phrases for user responses.
 * @author Lior Jigalo
 * @license MIT
 */

/**
 * Status codes for application operations.
 * @type {Object}
 */
const status = {
    ok: "OK",
    fail: "Fail",
    not_implemented: "Not Implemented"
}

/**
 * Time units used in the application.
 * @type {Object}
 */
const units ={
    second: 1000,
}

/**
 * List of phrases indicating user confirmation.
 * @type {string[]}
 */
const confirmationPhrases = [
    'im alert',
    'i am alert',
    'alert',
    'i am being alert',
    'i\'m alert',
    'i am fully alert',
    'alertness confirmed',
    'i\'m paying attention',
    'i am paying attention',
    'paying attention',
    'i\'m focused',
    'i am focused',
    'focused',
    'staying alert',
    'awake and alert',
    'i\'m awake',
    'i am awake',
    'i am here',
    'i\'m here',
    'here',
    'present and alert',
    'i\'m fully aware',
    'i am fully aware',
    'aware',
    'attentive',
    'i am attentive',
    'i\'m attentive',
    'on alert',
    'i am on alert',
    'i\'m on alert',
    'all good',
    'i\'m okay',
    'i\'m good',
    'i am okay',
    'okay',
    'everything is fine',
    'i\'m fine',
    'i am fine',
    'i am good',
    'i\'m good'
];

/**
 * List of phrases indicating no response from the user.
 * @type {string[]}
 */
const noResponsePhrases = [
    'BLANK_AUDIO',
    'clears throat'
];

/** Export constants */
module.exports = { status, units, confirmationPhrases, noResponsePhrases};