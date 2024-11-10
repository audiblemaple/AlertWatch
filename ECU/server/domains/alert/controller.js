const {currentDriveObject, updateDriveDataLog} = require("../../util/driveLogManager");
const Aplay = require('node-aplay');
const path = require('path');
const ffmpeg = require('fluent-ffmpeg');
const { exec } = require('child_process');
const {units} = require("../../util/const");

const updateDriveData = async (data) => {
	try {
		switch (data){
			case "high":
				currentDriveObject.high_alert_num += 1;
				break;

			case "mid":
				currentDriveObject.medium_alert_num += 1;
				break;

			case "low":
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

// Step 2: Gradually lower the volume of all sink inputs
const fadeOutVolume = async (sinkInputs, id, volume) => {
	for (let volume = 100; volume >= 0; volume -= 10) {
		for (const id of sinkInputs) {
			exec(`pactl set-sink-input-volume ${id} ${volume}%`);
		}
		await new Promise(resolve => setTimeout(resolve, 100)); // Wait 100ms between steps
	}
};

const setMuteVal = (sinkInputs, val) => {
	sinkInputs.forEach(id => {
		exec(`pacmd set-sink-input-mute ${id} ${val}`);
	});
}

const soundAlert = (fileName) => {
	const filePath = path.join(__dirname, '../../assets', fileName);

	// Step 1: Get all current sink input IDs and mute them
	exec("pacmd list-sink-inputs | grep index", (err, stdout) => {
		if (err) {
			console.error('Error retrieving sink inputs:', err);
		}

		// Extract all sink input IDs
		const sinkInputs = stdout
			.trim()
			.split('\n')
			.map(line => line.split(' ')[1])
			.filter(id => id);

		setMuteVal(sinkInputs, 1);

		// Step 2: Play the alert sound and get its duration
		ffmpeg.ffprobe(filePath, (err, metadata) => {
			if (err) {
				console.error('Error retrieving file metadata:', err);
				return;
			}

			const duration = metadata.format.duration * units.second;
			const player = new Aplay(filePath);
			player.play();

			// Step 3: Unmute all inputs after the alert sound finishes
			setTimeout(() => {
				player.pause();
				setMuteVal(sinkInputs, 0);
			}, duration);
		});
	});
};

// const soundAlert = (fileName) => {
// 	const filePath = path.join(__dirname, '../../assets', fileName);
//
// 	exec("pacmd list-sink-inputs | grep index | while read -r line; do pacmd set-sink-input-mute $(echo $line | cut -d' ' -f2) 1; done");
//
//
// 	// Get the duration of the audio file using fluent-ffmpeg
// 	ffmpeg.ffprobe(filePath, (err, metadata) => {
// 		if (err) {
// 			console.error('Error retrieving file metadata:', err);
// 			return;
// 		}
//
// 		// Get duration in seconds and convert to milliseconds
// 		const duration = metadata.format.duration * units.second;
//
// 		const player = new Aplay(filePath);
// 		player.play();
//
// 		let alertSinkId = alertSinkId.trim();
// 		muteOtherStreams(alertSinkId);
//
// 		// Stop playback after the duration of the file
// 		setTimeout(() => {
// 			player.pause();
//
// 			exec("pacmd list-sink-inputs | grep index | while read -r line; do pacmd set-sink-input-mute $(echo $line | cut -d' ' -f2) 0; done");
// 		}, duration);
// 	});
// };

module.exports = {updateDriveData, soundAlert}