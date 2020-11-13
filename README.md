# binary-brain
A densely connected neural net that has every neuron connected to every other neuron with binary (-1/+1) weights.  
It is different from typical neural networks in that no forward pass is defined. 
Instead, it uses the concept of cycles, where every cycle updates the neurons once.
This means the training algorithm is responsible for figuring out the way the input should propagate through the network, which makes this a very generic solution to machine learning tasks.

### Usage
Install rust - https://rustup.rs/   
If you want to run OpenAI gym examples, follow the install instructions for gym-rs - https://github.com/MrRobb/gym-rs
```
rustup override set nightly
cargo run --example <example>
```
For benchmarks:
```
rustup override set nightly
RUSTFLAGS="-C target-cpu=native" cargo bench
```

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