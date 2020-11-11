# binary-brain
A densely connected neural net that has every neuron connected to every other neuron with binary (-1/+1) weights.

### Compiling
```
$ rustup override set nightly
$ RUSTFLAGS="-C target-cpu=native" cargo <bench|run --example <example>>
```
Requires the nightly compiler only for benchmarks, which is the primary focus as of writing.
Setting target-cpu=native greatly improves performance on modern cpus (only tested on x86_64)
