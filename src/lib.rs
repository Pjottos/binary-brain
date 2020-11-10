#![feature(test)]

extern crate static_assertions as sa;
extern crate test;

#[macro_export]
macro_rules! binary_nn {
    ($Name:ident, $InputCount:expr, $HiddenCount:expr, $OutputCount:expr) => {
        sa::const_assert_eq!($InputCount % 32, 0);
        sa::const_assert_eq!($HiddenCount % 32, 0);
        sa::const_assert_eq!($OutputCount % 32, 0);

        struct $Name {
            weights: [u32; Self::WEIGHT_COUNT / 32],
            values: [u32; Self::WEIGHT_COUNT / 32],
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
                        ($InputCount * $HiddenCount) / (32 / 8)
                    ));
                    rng.fill_bytes(&mut result.act);

                    result
                }
            }

            #[inline]
            pub fn cycle(&mut self, input: &[u8; $InputCount]) -> [bool; $OutputCount]{
                let mut output = [false; $OutputCount];

                for i in 0..Self::NEURON_COUNT {
                    let mut sum = self.calc_sum(i);
                    if i < $InputCount {
                        sum += input[i] as i32;
                    }
                    
                    let fire = sum >= self.act[i] as i32;
                    for j in 0..Self::NEURON_COUNT {
                        let idx = j * (Self::NEURON_COUNT / 32) + i / 32;
                        let mask = 1 << (i % 32);

                        // TODO: roughly 98% of cycles is spent on the 'or' and 'xor' here, find a way to optimize it
                        // it seems to be caused by the memory writes

                        // this makes it so the 'or' is never done if not needed
                        // since the loop is unrolled, there is no overhead from the comparison
                        if fire {
                            self.values[idx] |= mask;
                        } else {
                            self.values[idx] ^= mask;
                        }
                    }

                    let output_start = Self::NEURON_COUNT - $OutputCount;
                    if i > output_start && fire {
                        let idx = i - output_start;
                        output[idx] = true;
                    }
                }
                
                output
            }

            #[inline]
            fn calc_sum(&self, neuron: usize) -> i32 {
                let mut sum = 0;
                for j in 0..Self::NEURON_COUNT / 32 {
                    let idx = (neuron * Self::NEURON_COUNT) / 32 + j;
                    sum += self.values[idx].count_ones() as i32;
                    sum -= self.values[idx].count_zeros() as i32;
                }

                sum
            }
        }
    };
}

#[cfg(test)]
mod tests {
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