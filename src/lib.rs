#![feature(test)]

extern crate test;

use std::iter::repeat_with;
use rand_xoshiro::rand_core::{RngCore};

pub struct BinaryBrain {
    weights: Vec<u64>,
    values: Vec<u64>,
    act: Vec<u8>,
    input_count: usize,
    output_count: usize,
}

impl BinaryBrain {
    pub fn new(input_count: usize, output_count: usize, total_count: usize, rng: &mut impl RngCore) -> BinaryBrain {
        let mut act = vec![0; total_count];
        rng.fill_bytes(act.as_mut_slice());

        let weight_count = total_count * total_count;

        BinaryBrain {
            weights: repeat_with(|| rng.next_u64()).take(weight_count / 64).collect(),
            values: repeat_with(|| rng.next_u64()).take(weight_count / 64).collect(),
            act: act,
            input_count: input_count,
            output_count: output_count,
        }
    }

    #[inline]
    pub fn cycle(&mut self, input: &[u8]) -> Vec<(bool, i32)> {
        let mut output = vec![(false, 0); self.output_count];
        let neuron_count = self.act.len();
        for i in (0..neuron_count).step_by(64) {
            // write the values of each connection in batches to reduce memory i/o
            // this provides a dramatic speedup of about 133x over setting each bit individually
            // (tested with a 64x4096x64 network)
            let mut batch: u64 = 0;

            for j in i..i + 64 {
                let mut sum = self.calc_sum(j);
                
                if j < self.input_count {
                    sum += input[j] as i32;
                } 
                let fire = sum >= self.act[j] as i32;
                // set bits in the batch variable in the order it should have 
                // in the value matrix 
                batch |= (fire as u64) << (j - i);
                
                let output_start = neuron_count - self.output_count;
                if j > output_start {
                    let idx = j - output_start;
                    output[idx] = (fire, sum);
                }
            }
            
            for j in (0..neuron_count).step_by(64) {
                let idx = j * (neuron_count / 64) + i / 64;
                self.values[idx] = batch;
            }
        }
        
        output
    }

    #[inline]
    fn calc_sum(&self, neuron: usize) -> i32 {
        let mut sum = 0;
        let weight_row_size = self.act.len() / 64;
        let start = neuron * weight_row_size;

        for j in start..start + weight_row_size {
            // perform an xnor with the weights and values, then add the popcount of that value
            // and subtract the zero count
            // this produces the following value per connection/weight:
            //          !weight   weight  
            // !value |    1    |  -1    |
            // -------+---------+--------+
            // value  |   -1    |   1    |

            // couldn't get the bounds check optimized away unfortunately
            let weighted = unsafe { !(self.weights.get_unchecked(j) ^ self.values.get_unchecked(j)) };
            sum += weighted.count_ones() as i32;
            sum -= weighted.count_zeros() as i32;
        }

        sum
    }
}

#[cfg(test)]
mod benches {
    use test::{Bencher};
    use rand_xoshiro::rand_core::{SeedableRng};
    use rand_xoshiro::Xoshiro256StarStar;
    use crate::BinaryBrain;


    #[bench]
    fn cycle_64_64_64(b: &mut Bencher) {
        let mut rng = Xoshiro256StarStar::seed_from_u64(1234);
        let mut nn = BinaryBrain::new(64, 64, 192, &mut rng);
        let input = [u8::MAX; 64];

        b.iter(|| {
            nn.cycle(&input)
        });
    } 

    #[bench]
    fn cycle_64_512_64(b: &mut Bencher) {
        let mut rng = Xoshiro256StarStar::seed_from_u64(1234);
        let mut nn = BinaryBrain::new(64, 64, 640, &mut rng);
        let input = [u8::MAX; 64];

        b.iter(|| {
            nn.cycle(&input)
        });
    }

    #[bench]
    fn cycle_64_4096_64(b: &mut Bencher) {
        let mut rng = Xoshiro256StarStar::seed_from_u64(1234);
        let mut nn = BinaryBrain::new(64, 64, 4224, &mut rng);
        let input = [u8::MAX; 64];

        b.iter(|| {
            nn.cycle(&input)
        });
    }

    // #[bench]
    // fn cycle_64_131072_64(b: &mut Bencher) {
    //     let mut rng = Xoshiro256StarStar::seed_from_u64(1234);
    //     let mut nn = BinaryBrain::new(64, 64, 131200, &mut rng);
    //     let input = [u8::MAX; 64];

    //     b.iter(|| {
    //         nn.cycle(&input)
    //     });
    // }
}