const app = require("./app");
require("dotenv").config();
const {PORT, IP} = process.env;
const {} = require("./util/driveLogManager");
const { initWebSocket } = require("./websocket");
const {soundAlert} = require("./domains/alert/controller");

const startApp = () => {
    // Create HTTP server and listen on 0.0.0.0 and specified PORT
    const server = app.listen(PORT, `${IP}`, () => {
        console.log(`Server running on port ${PORT} and listening on 0.0.0.0`);
    });

    console.log("sounding file");
    soundAlert("attention test.wav");

    // Initialize WebSocket server on the same HTTP server
    initWebSocket(server);
};

startApp();