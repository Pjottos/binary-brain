#![feature(test)]

extern crate static_assertions as sa;
extern crate test;

#[macro_export]
macro_rules! binary_nn {
    ($Name:ident, $InputCount:expr, $HiddenCount:expr, $OutputCount:expr, $MaxInputAct:expr) => {
        sa::const_assert_eq!($InputCount % 64, 0);
        sa::const_assert_eq!($HiddenCount % 64, 0);
        sa::const_assert_eq!($OutputCount % 64, 0);

        sa::const_assert!($InputCount + $HiddenCount - 1 <= u16::MAX);
        sa::const_assert!($MaxInputAct <= u16::MAX);
        struct $Name {
            input_weights: [u64; ($InputCount * $HiddenCount) / 64],
            input_act: [u16; $InputCount],
            input_hidden_v: [u64; ($InputCount * $HiddenCount) / 64],

            hidden_weights: [u64; ($HiddenCount * $HiddenCount) / 64],
            hidden_act: [u16; $HiddenCount],
            hidden_hidden_v: [u64; ($HiddenCount * $HiddenCount) / 64],

            output_weights: [u64; ($HiddenCount * $OutputCount) / 64],
            output_act: [u16; $OutputCount],
            hidden_output_v: [u64; ($HiddenCount * $OutputCount) / 64],

            output: [bool; $OutputCount],
        }

        impl $Name {
            const MAX_ACT_HIDDEN: u16 = $InputCount + $HiddenCount - 1;
            const MAX_ACT_OUTPUT: u16 = $HiddenCount;

            pub fn new(rng: &mut impl RngCore) -> $Name {
                unsafe {
                    let mut result = $Name {
                        input_weights: MaybeUninit::uninit().assume_init(),
                        input_act: MaybeUninit::uninit().assume_init(),
                        input_hidden_v: MaybeUninit::zeroed().assume_init(),

                        hidden_weights: MaybeUninit::uninit().assume_init(),
                        hidden_act: MaybeUninit::uninit().assume_init(),
                        hidden_hidden_v: MaybeUninit::zeroed().assume_init(),

                        output_weights: MaybeUninit::uninit().assume_init(),
                        output_act: MaybeUninit::uninit().assume_init(),
                        hidden_output_v: MaybeUninit::zeroed().assume_init(),

                        output: [false; $OutputCount],
                    };
    
                    rng.fill_bytes(std::slice::from_raw_parts_mut(
                        result.input_weights.as_mut_ptr() as *mut u8,
                        ($InputCount * $HiddenCount) / 8
                    ));
                    rng.fill_bytes(std::slice::from_raw_parts_mut(
                        result.input_act.as_mut_ptr() as *mut u8,
                        $InputCount * 2
                    ));

                    rng.fill_bytes(std::slice::from_raw_parts_mut(
                        result.hidden_weights.as_mut_ptr() as *mut u8,
                        ($HiddenCount * $HiddenCount) / 8
                    ));
                    rng.fill_bytes(std::slice::from_raw_parts_mut(
                        result.hidden_act.as_mut_ptr() as *mut u8,
                        $HiddenCount * 2
                    ));

                    rng.fill_bytes(std::slice::from_raw_parts_mut(
                        result.output_weights.as_mut_ptr() as *mut u8,
                        ($HiddenCount * $OutputCount) / 8
                    ));
                    rng.fill_bytes(std::slice::from_raw_parts_mut(
                        result.output_act.as_mut_ptr() as *mut u8,
                        $OutputCount * 2
                    ));
    
                    result
                }
            }

            #[inline]
            pub fn cycle(&mut self, input: &[u16; $InputCount]) {
                for i in 0..$InputCount {
                    let fire = input[i] >= self.input_act[i];
                    
                    for j in 0..$HiddenCount {
                        // go to the right row in the bit matrix, then divide by 64 to get to the first u64 of the row
                        // then, go to the right u64 by adding i / 64
                        let idx = (j * $InputCount) / 64 + i / 64;
                        let mask = 1 << (i % 64);
                        if fire {
                            self.input_hidden_v[idx] |= mask;
                        } else {
                            self.input_hidden_v[idx] &= !mask;
                        }
                    }
                }

                for i in 0..$HiddenCount {
                    let mut sum = 0;
                    for j in 0..$InputCount / 64 {
                        // see above
                        let idx = (i * $InputCount) / 64 + j;
                        sum += self.input_hidden_v[idx].count_ones();
                    }

                    for j in 0..$HiddenCount / 64 {
                        let idx = (i * $HiddenCount) / 64 + j;
                        sum += self.hidden_hidden_v[idx].count_ones();
                    }

                    let fire = sum >= self.hidden_act[i] as u32;
                    for j in 0..$HiddenCount {
                        let idx = (j * $HiddenCount) / 64 + i / 64;
                        let mask = 1 << (i % 64);
                        if fire {
                            self.hidden_hidden_v[idx] |= mask;
                        } else {
                            self.hidden_hidden_v[idx] &= !mask;
                        }
                    }

                    for j in 0..$OutputCount {
                        let idx = (j * $HiddenCount) / 64 + i / 64;
                        let mask = 1 << (i % 64);
                        if fire {
                            self.hidden_output_v[idx] |= mask;
                        } else {
                            self.hidden_output_v[idx] &= !mask;
                        }
                    }
                }

                for i in 0..$OutputCount {
                    let mut sum = 0;
                    for j in 0..$HiddenCount / 64 {
                        let idx = (i * $HiddenCount) / 64 + j;
                        sum += self.hidden_hidden_v[idx].count_ones();
                    }

                    self.output[i] = sum >= self.output_act[i] as u32;
                }
            }

            #[inline]
            pub fn output(&self) -> &[bool; $OutputCount] {
                &self.output
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

        binary_nn!(Net, 64, 64, 64, u16::MAX); 

        #[bench]
        fn cycle(b: &mut Bencher) {
            let mut rng = Xoshiro256StarStar::from_seed([0u8; 32]);
            let mut nn = Net::new(&mut rng);
            let input = [u16::MAX; 64];

            b.iter(|| {
                nn.cycle(&input)
            });
        }
    }

    mod nn_64_512_64 {
        use super::*;

        binary_nn!(Net, 64, 512, 64, u16::MAX); 

        #[bench]
        fn cycle(b: &mut Bencher) {
            let mut rng = Xoshiro256StarStar::from_seed([0u8; 32]);
            let mut nn = Net::new(&mut rng);
            let input = [u16::MAX; 64];

            b.iter(|| {
                nn.cycle(&input)
            });
        }
    }

    mod nn_64_4096_64 {
        use super::*;

        binary_nn!(Net, 64, 4096, 64, u16::MAX); 

        #[bench]
        fn cycle(b: &mut Bencher) {
            let mut rng = Xoshiro256StarStar::from_seed([0u8; 32]);
            let mut nn = Net::new(&mut rng);
            let input = [u16::MAX; 64];

            b.iter(|| {
                nn.cycle(&input)
            });
        }
    }

    
}