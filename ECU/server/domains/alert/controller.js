const {currentDriveObject, updateDriveDataLog} = require("../../util/driveLogManager");
const updateDriveData = async (data) => {
	try {
		let actions = {
			needs_alert: null,
			needs_response: null
		}

		actions.needs_alert = true;
		actions.needs_response = true;
		actions.needs_response = true;

		switch (data){
			case "high":
				currentDriveObject.high_alert_num += 1;
				break;

			case "mid":
				currentDriveObject.medium_alert_num += 1;
				break;

			case "low":
				actions.needs_response = false;
				currentDriveObject.mild_alert_num += 1;
				break;
			default:
				console.log(currentDriveObject)
				throw Error('Error processing alert!');
		}

		currentDriveObject.alert_count = currentDriveObject.high_alert_num + currentDriveObject.medium_alert_num + currentDriveObject.mild_alert_num;
		updateDriveDataLog()

	} catch (error){
		throw error;
	}
}

module.exports = {updateDriveData}