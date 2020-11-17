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
test benches::cycle_32768 ... bench:  11,015,509 ns/iter (+/- 1,051,552)
test benches::cycle_4096  ... bench:     133,047 ns/iter (+/- 6,552)
test benches::cycle_512   ... bench:       4,894 ns/iter (+/- 601)
test benches::cycle_64    ... bench:         229 ns/iter (+/- 11)
```
The number after `cycle` is the amount of neurons in the brain.
For each benchmark 32 inputs and 32 outputs are used.