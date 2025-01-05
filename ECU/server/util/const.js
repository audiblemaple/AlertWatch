const status = {
    ok: "OK",
    fail: "Fail",
    not_implemented: "Not Implemented"
}

const units ={
    second: 1000,
}

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
    'i am okay',
    'okay',
    'everything is fine',
    'i\'m fine',
    'i am fine',
    'i am good',
    'i\'m good'
];

const noResponsePhrases = [
    'BLANK_AUDIO',
    'clears throat'
];

module.exports = { status, units, confirmationPhrases, noResponsePhrases};