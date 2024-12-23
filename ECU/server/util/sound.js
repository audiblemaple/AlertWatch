const { exec } = require('child_process');

// Function to mute all streams
function muteAllStreams() {
  exec("pactl list sink-inputs | grep -oP '(?<=Sink Input #)\\d+'", (err, stdout) => {
    if (err) {
      console.error(`Error listing streams: ${err}`);
      return;
    }
    const streamIds = stdout.trim().split('\n');
    streamIds.forEach(streamId => {
      exec(`pactl set-sink-input-mute ${streamId} 1`);
    });
  });
}

// Function to unmute all streams
function unmuteAllStreams() {
  exec("pactl list sink-inputs | grep -oP '(?<=Sink Input #)\\d+'", (err, stdout) => {
    if (err) {
      console.error(`Error listing streams: ${err}`);
      return;
    }
    const streamIds = stdout.trim().split('\n');
    streamIds.forEach(streamId => {
      exec(`pactl set-sink-input-mute ${streamId} 0`);
    });
  });
}

// Function to play custom sound
function playSound(filePath) {
  exec(`paplay ${filePath}`, (err) => {
    if (err) {
      console.error(`Error playing sound: ${err}`);
    }
  });
}
