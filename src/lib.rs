#![feature(test)]

extern crate static_assertions as sa;
extern crate test;

#[macro_export]
macro_rules! binary_nn {
    ($Name:ident, $InputCount:expr, $HiddenCount:expr, $OutputCount:expr) => {
        // it's possible to use u32 to provide more flexibility for input/output count
        // but that results in roughly a 2x slowdown
        sa::const_assert_eq!($InputCount % 64, 0);
        sa::const_assert_eq!($HiddenCount % 64, 0);
        sa::const_assert_eq!($OutputCount % 64, 0);

        struct $Name {
            weights: [u64; Self::WEIGHT_COUNT / 64],
            values: [u64; Self::WEIGHT_COUNT / 64],
            act: [u8; Self::NEURON_COUNT],
        }

        impl $Name {
            const NEURON_COUNT: usize = $InputCount + $HiddenCount + $OutputCount;
            const WEIGHT_COUNT: usize = Self::NEURON_COUNT * Self::NEURON_COUNT;

            pub fn new(rng: &mut impl RngCore) -> $Name {
                unsafe {
                    let mut result = $Name {
                        weights: MaybeUninit::uninit().assume_init(),
                        act: MaybeUninit::uninit().assume_init(),
                        values: MaybeUninit::zeroed().assume_init(),
                    };
    
                    rng.fill_bytes(std::slice::from_raw_parts_mut(
                        result.weights.as_mut_ptr() as *mut u8,
                        ($InputCount * $HiddenCount) / (64 / 8)
                    ));
                    rng.fill_bytes(&mut result.act);

                    result
                }
            }

            #[inline]
            pub fn cycle(&mut self, input: &[u8; $InputCount]) -> [bool; $OutputCount] {
                let mut output = [false; $OutputCount];

                for i in (0..Self::NEURON_COUNT).step_by(64) {
                    // write the values of each connection in batches to reduce memory i/o
                    // this provides a dramatic speedup of about 133x over setting each bit individually
                    // (tested with a 64x4096x64 network)
                    let mut batch: u64 = 0;

                    for j in i..i + 64 {
                        let mut sum = self.calc_sum(j);
                        if j < $InputCount {
                            sum += input[j] as i32;
                        }    
                        let fire = sum >= self.act[j] as i32;
                        // set bits in the batch variable in the order it should have 
                        // in the value matrix 
                        batch |= (fire as u64) << (j - i);
                        
                        let output_start = Self::NEURON_COUNT - $OutputCount;
                        if j > output_start && fire {
                            let idx = j - output_start;
                            output[idx] = true;
                        }
                    }
                    
                    for j in (0..Self::NEURON_COUNT).step_by(64) {
                        let idx = j * (Self::NEURON_COUNT / 64) + i / 64;

                        self.values[idx] = batch;
                    }
                }
                
                output
            }

            #[inline]
            fn calc_sum(&self, neuron: usize) -> i32 {
                let mut sum = 0;
                for j in 0..Self::NEURON_COUNT / 64 {
                    let idx = neuron * (Self::NEURON_COUNT / 64) + j;
                    // perform an xnor with the weights and values, then add the popcount of that value
                    // and subtract the zero count
                    // this produces the following value per connection/weight:
                    //          !weight   weight  
                    // !value |    1    |  -1    |
                    // -------+---------+--------+
                    // value  |   -1    |   1    |

                    let weighted = !(self.weights[idx] ^ self.values[idx]);
                    sum += weighted.count_ones() as i32;
                    sum -= weighted.count_zeros() as i32;
                }

                sum
            }
        }
    };
}

#[cfg(test)]
mod benches {
    use test::{Bencher};
    use rand_xoshiro::rand_core::{SeedableRng, RngCore};
    use rand_xoshiro::Xoshiro256StarStar;
    use std::mem::MaybeUninit;
    
    
    mod nn_64_64_64 {
        use super::*;

        binary_nn!(Net, 64, 64, 64); 

        #[bench]
        fn cycle(b: &mut Bencher) {
            let mut rng = Xoshiro256StarStar::seed_from_u64(1234);
            let mut nn = Net::new(&mut rng);
            let input = [u8::MAX; 64];

            b.iter(|| {
                nn.cycle(&input)
            });
        }
    }

    mod nn_64_512_64 {
        use super::*;

        binary_nn!(Net, 64, 512, 64); 

        #[bench]
        fn cycle(b: &mut Bencher) {
            let mut rng = Xoshiro256StarStar::seed_from_u64(1234);
            let mut nn = Net::new(&mut rng);
            let input = [u8::MAX; 64];

            b.iter(|| {
                nn.cycle(&input)
            });
        }
    }

    mod nn_64_4096_64 {
        use super::*;

        binary_nn!(Net, 64, 4096, 64); 

        #[bench]
        fn cycle(b: &mut Bencher) {
            let mut rng = Xoshiro256StarStar::seed_from_u64(1234);
            let mut nn = Net::new(&mut rng);
            let input = [u8::MAX; 64];

            b.iter(|| {
                nn.cycle(&input)
            });
        }
    }
}