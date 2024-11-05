const express = require("express");
const router = express.Router();
const { status } = require('../../util/const');
const {updateDriveData} = require("./controller");

// route: /api/v1/alert
router.post("/", async (req, res) => {
	try {
		let { severity } = req.body;

		await updateDriveData(severity);

		if (severity === "low")
			// TODO: sound alert to do a break if you're tired
			res.status(200).json({status: status.ok, message: "Alert sounded"});

		// TODO: parsing command: ./main --output-json --threads 10 --model models/ggml-tiny.en.bin --file ../output.wav

		// TODO: Mute multimedia
		// TODO: Sound alert
		// TODO: Listen for a response
		// TODO: Check response
			// TODO: Response good (*)
				// TODO: unmute multimedia
				// TODO: sound alert to do a break if you're tired
				// TODO: return OK
			// TODO: Response bad
				// TODO: Wait for a good response twice
				// TODO: Response good -> go to (*)
				// TODO: Sound the deceleration alert
				// TODO: Wait 2 seconds, alert for deceleration start decelerating

		res.status(200).json({status: status.not_implemented, message: "NOT implemented"});

	} catch (error) {
		res.status(400).json({status: status.fail, message: error.message});
	}
});

module.exports = router;