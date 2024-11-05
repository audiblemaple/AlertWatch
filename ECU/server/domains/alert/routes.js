const express = require("express");
const router = express.Router();
const { failure, ok, not_implemented } = require('../../util/const');
const {updateDriveData} = require("./controller");

// // private route
// router.get("/private_route", auth, (req, res) => {
// 	res.status(200).send(`you are in the private territory of ${req.currentUser.email}`);
// });

// route: /api/v[1-9]+/alert
router.post("/", async (req, res) => {
	try {
		let { severity } = req.body;

		await updateDriveData(severity);



		res.status(200).json({status: not_implemented, message: "NOT implemented"});

	} catch (error) {
		res.status(400).json({status: failure, message: error.message});
	}
});

module.exports = router;