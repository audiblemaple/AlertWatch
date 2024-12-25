const { exec, execFile } = require('child_process');
const path = require("path");
const { spawn } = require('child_process');

const ffmpeg = require('fluent-ffmpeg');
const natural = require('natural');
const stemmer = natural.PorterStemmer; // Using Porter Stemmer
const tokenizer = new natural.WordTokenizer();


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

/**
 * Plays a sound file.
 * @param {string} filePath - Path to the WAV file to play.
 * @returns {Promise<void>} - Resolves when the sound has finished playing.
 */
function playSound(filePath) {
    return new Promise((resolve, reject) => {
        muteAllStreams();
        const player = spawn('aplay', [filePath]);

        player.on('error', (err) => {
            console.error(`Error playing sound: ${err.message}`);
            reject(err);
        });

        player.on('close', (code) => {
            if (code !== 0) {
                reject(new Error(`Sound player exited with code ${code}`));
            } else {
                resolve();
            }
        });
        setTimeout(unmuteAllStreams, 4000);
    });
}



const confirmationPhrases = [
    'im alert',
    'i am alert',
    'alert',
    'i am being alert',
    'i\'m alert',
    'i am fully alert',
    'alertness confirmed'
    // Add more variations as needed
];

// Preprocess confirmation phrases with stemming
const stemmedConfirmationPhrases = confirmationPhrases.map(phrase => {
    return phrase.split(' ').map(word => stemmer.stem(word)).join(' ');
});

/**
 * Checks if the transcription includes a confirmation of alertness.
 * @param {Array<string>} lines - Parsed transcription lines.
 * @returns {boolean} - True if confirmation is detected, else false.
 */
function hasAlertConfirmation(lines) {
    for (const line of lines) {
        // Tokenize the line into words
        const words = tokenizer.tokenize(line.toLowerCase());

        // Stem each word
        const stemmedWords = words.map(word => stemmer.stem(word));

        // Reconstruct the line from stemmed words
        const stemmedLine = stemmedWords.join(' ');

        // Check against each stemmed confirmation phrase
        for (const phrase of stemmedConfirmationPhrases) {
            if (stemmedLine.includes(phrase)) {
                return true;
            }
        }
    }
    return false;
}

/**
 * Transcribes a WAV file using whisper-cli, capturing stdout/stderr streams.
 * @param {string} audioFilePath - The path to the .wav file
 * @returns {Promise<string>} - The raw transcription output from whisper-cli
 */
async function transcribeWithWhisper(audioFilePath) {
  // Path to the whisper-cli binary
  const whisperCliPath = path.join(__dirname, "../assets/bin", "whisper-cli");
  // Path to the whisper model
  const modelPath = path.join(__dirname, "../assets/models", "ggml-tiny.en.bin");

  // Build arguments
  const args = ["--model", modelPath, "--no-prints", audioFilePath];

  return new Promise((resolve, reject) => {
    // We do NOT pass a callback here, because we want manual streaming.
    const child = execFile(whisperCliPath, args, { shell: false });

    let stdoutData = "";
    let stderrData = "";

    child.stdout.on("data", (data) => {
      stdoutData += data.toString();
    });

    child.stderr.on("data", (data) => {
      stderrData += data.toString();
    });

    child.on("close", (code) => {
      if (code !== 0) {
        return reject(
          new Error(
            `whisper-cli failed with exit code ${code}.\nStderr: ${stderrData}`
          )
        );
      }
      // Return the entire raw transcription output
      resolve(stdoutData.trim());
    });

    // If an error is emitted at the process level, catch it
    child.on("error", (err) => {
      reject(err);
    });
  });
}




// A helper function to parse the raw transcription lines
function parseWhisperOutput(rawOutput) {
  // Each line might look like:
  // [00:00:00.000 --> 00:00:04.000]   alertness check...
  // We'll remove that bracketed part using a regex and just keep the text portion.
  const lines = rawOutput.split("\n").filter(Boolean);

  return lines.map((line) =>
    line.replace(/^\[[^\]]*\]\s*/, "").trim()
  );
}

/**
 * Records audio using ffmpeg and saves it as a WAV file in 16kHz mono format.
 * @param {string} outputFilePath - Path to save the recorded WAV file.
 * @param {number} durationInSeconds - Duration to record in seconds.
 * @returns {Promise<void>}
 */
function recordAudioWithFFmpeg(outputFilePath, durationInSeconds = 5) {
    return new Promise((resolve, reject) => {

        ffmpeg()
            .input('default') // Adjust based on OS. 'default' works for Linux with ALSA.
            .inputFormat('alsa') // 'alsa' for Linux. Use 'avfoundation' for macOS, 'dshow' for Windows.
            .audioFrequency(16000)
            .audioChannels(1)
            .audioCodec('pcm_s16le')
            .format('wav')
            .duration(durationInSeconds)
            .on('end', () => {
                resolve();
            })
            .on('error', (err) => {
                reject(err);
            })
            .save(outputFilePath);
    });
}



module.exports = {
    playSound,
    transcribeWithWhisper,
    parseWhisperOutput,
//  recordAudio
    recordAudioWithFFmpeg,
    hasAlertConfirmation
};