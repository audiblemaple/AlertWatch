const app = require("./app");
require("dotenv").config();
const {PORT, IP} = process.env;
const {} = require("./util/driveLogManager");
const { initWebSocket } = require("./websocket");

const startApp = () => {
    // Create HTTP server and listen on 0.0.0.0 and specified PORT
    const server = app.listen(PORT, "0.0.0.0", () => {
        console.log(`Server running on port ${PORT} and listening on 0.0.0.0`);
    });

    // Initialize WebSocket server on the same HTTP server
    initWebSocket(server);
};

startApp();