# binary-brain
A densely connected neural net that has every neuron connected to every other neuron with binary (-1/+1) weights.

### Usage
Currently, only the evaluation logic is implemented and the brain cannot be trained yet.
To benchmark the evaluation:
```
rustup override set nightly
RUSTFLAGS="-C target-cpu=native" cargo bench
```
Setting target-cpu=native greatly improves performance on modern cpus (only tested on x86_64)

To see a random brain in action:
```
rustup override set nightly
RUSTFLAGS="-C target-cpu=native" cargo run --example <example>
```

This project is not meant to be used as a library but it is possible.

### Performance
Benchmark results on an i7-4790k:
```
test benches::cycle_32768 ... bench:  11,466,698 ns/iter (+/- 329,784)
test benches::cycle_4096  ... bench:     162,130 ns/iter (+/- 7,157)
test benches::cycle_512   ... bench:       4,185 ns/iter (+/- 171)
test benches::cycle_64    ... bench:         216 ns/iter (+/- 11)
```
The number after `cycle` is the amount of neurons in the brain.
For each benchmark 32 inputs and 32 outputs are used.