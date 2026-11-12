// JSON-lines bridge: original ConvNetJS training, unchanged network_basic.js.
const fs = require('fs');
const vm = require('vm');
const readline = require('readline');
const path = require('path');
const [backend, starter, seedText] = process.argv.slice(2);
let seed = Number(seedText) >>> 0;
// Explicit reproducibility choice replacing the browser's unseeded Math.random.
const math = Object.create(Math);
math.random = () => {
  seed = (Math.imul(1664525, seed) + 1013904223) >>> 0;
  return seed / 4294967296;
};
const ctx = vm.createContext({Math: math, console: {log: (...x) => console.error(...x)},
  draw_net: () => {}, draw_stats: () => {}});
ctx.window = ctx;
for (const name of ['convnet.js', 'util.js', 'deepqlearn.js']) {
  vm.runInContext(fs.readFileSync(path.join(backend, 'reference/original_js/convnetjs', name), 'utf8'), ctx);
}
vm.runInContext(fs.readFileSync(starter, 'utf8'), ctx);
const lines = readline.createInterface({input: process.stdin});
lines.on('line', line => {
  const req = JSON.parse(line);
  let result;
  if (req.op === 'learn') result = ctx.learn(req.state, req.reward);
  else if (req.op === 'seed') { ctx.brain.reset_seed(req.seed); result = true; }
  else if (req.op === 'export') result = {net: ctx.brain.value_net.toJSON(),
    age: ctx.brain.age, epsilon: ctx.brain.epsilon, replay_size: ctx.brain.experience.length};
  else if (req.op === 'q') {
    const vol = new ctx.convnetjs.Vol(1, 1, 30);
    vol.w = req.state;
    result = Array.from(ctx.brain.value_net.forward(vol).w);
  } else throw new Error('Unknown operation');
  process.stdout.write(JSON.stringify(result) + '\n');
});
