// Laeuft im Web Worker des Pools: Whisper-Transkription eines Audio-Abschnitts.
importScripts('/whisper-task.js');
registerProject('musicai', (task, ctx) => self.runWhisperTask(task, ctx));
