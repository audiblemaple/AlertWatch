let locks = {
    alert_lock: false
}

let carState = {
    accelerating: false,
    decelerating: false,
    stopped: false,
    canceled: false
}

module.exports = {locks, carState}