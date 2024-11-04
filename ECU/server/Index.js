const app = require("./app");
require("dotenv").config();
const {PORT, IP} = process.env;
const {} = require("./util/driveLogManager");

const startApp = () => {
	app.listen(PORT, IP, () => {
		console.log(`Server running on port ${PORT} and listening on ${IP}.`)
	});
};

// console.log(`Using log file: ${logFileName}`);
startApp();