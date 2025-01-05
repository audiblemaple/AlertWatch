const path = require("path");
let locks = {
    alert_lock: false
}

let carState = {
    accelerating: true,
    decelerating: false,
    cruising: false,
    stopped: false,
    canceled: false
}

const sounds = {
    takeABreak:         path.join(__dirname, '../assets/sounds', "takeABreak.wav"),
    attentionTest:      path.join(__dirname, '../assets/sounds', 'attentionTest.wav'),
    failedToParse:      path.join(__dirname, '../assets/sounds', 'failedToParse.wav'),
    noResponse:         path.join(__dirname, '../assets/sounds', 'noResponse.wav'),
    decelerateWarning:  path.join(__dirname, '../assets/sounds', 'decelerateWarning.wav'),
    decelerating:       path.join(__dirname, '../assets/sounds', 'decelerating.wav'),
};

const user_status = {
    noResponse: 0,
    userResponded: 1,
    failedToParse: 2
}

module.exports = {locks, carState, user_status, sounds}