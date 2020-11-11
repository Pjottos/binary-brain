#![feature(test)]

extern crate test;

use std::iter::repeat_with;
use rand_xoshiro::rand_core::{RngCore};

pub type Result<T> = std::result::Result<T, BinaryBrainError>;

pub struct BinaryBrain {
    weight_value_store: Vec<u64>,
    act: Vec<u8>,
    input_count: usize,
    output_count: usize,
}

impl BinaryBrain {
    pub fn new(input_count: usize, output_count: usize, total_count: usize, rng: &mut impl RngCore) -> Result<BinaryBrain> {
        if total_count % 64 != 0 {
            return Err(BinaryBrainError::TotalNotDivisbleBy64);
        }
        if input_count + output_count > total_count {
            return Err(BinaryBrainError::InputOutputAboveTotal);
        }

        let mut act = vec![0; total_count];
        rng.fill_bytes(act.as_mut_slice());

        let weight_count = total_count * total_count;

        Ok(BinaryBrain {
            weight_value_store: repeat_with(|| rng.next_u64()).take((weight_count * 2) / 64).collect(),
            act: act,
            input_count: input_count,
            output_count: output_count,
        })
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
                // index will definitely be within bounds
                let mut sum = unsafe { self.calc_sum(j) };
                
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
                self.set_value(idx, batch);
            }
        }
        
        output
    }

    #[inline]
    unsafe fn calc_sum(&self, neuron: usize) -> i32 {
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
            let weighted = !(self.weight_unchecked(j) ^ self.value_unchecked(j));
            sum += weighted.count_ones() as i32;
            sum -= weighted.count_zeros() as i32;
        }

        sum
    }

    #[inline]
    fn value(&self, idx: usize) -> u64 {
        self.weight_value_store[idx * 2 + 1]
    }

    #[inline]
    fn weight(&self, idx: usize) -> u64 {
        self.weight_value_store[idx * 2]
    }

    #[inline]
    fn set_value(&mut self, idx: usize, value: u64) {
        self.weight_value_store[idx * 2 + 1] = value;
    }

    #[inline]
    fn set_weight(&mut self, idx: usize, value: u64) {
        self.weight_value_store[idx * 2] = value;
    }

    #[inline]
    unsafe fn value_unchecked(&self, idx: usize) -> &u64 {
        self.weight_value_store.get_unchecked(idx * 2 + 1)
    }

    #[inline]
    unsafe fn weight_unchecked(&self, idx: usize) -> &u64 {
        self.weight_value_store.get_unchecked(idx * 2)
    }
}

#[derive(Debug)]
enum BinaryBrainError {
    TotalNotDivisbleBy64,
    InputOutputAboveTotal,
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
        let mut nn = BinaryBrain::new(64, 64, 192, &mut rng).unwrap();
        let input = [u8::MAX; 64];

        b.iter(|| {
            nn.cycle(&input)
        });
    } 

    #[bench]
    fn cycle_64_512_64(b: &mut Bencher) {
        let mut rng = Xoshiro256StarStar::seed_from_u64(1234);
        let mut nn = BinaryBrain::new(64, 64, 640, &mut rng).unwrap();
        let input = [u8::MAX; 64];

        b.iter(|| {
            nn.cycle(&input)
        });
    }

    #[bench]
    fn cycle_64_4096_64(b: &mut Bencher) {
        let mut rng = Xoshiro256StarStar::seed_from_u64(1234);
        let mut nn = BinaryBrain::new(64, 64, 4224, &mut rng).unwrap();
        let input = [u8::MAX; 64];

        b.iter(|| {
            nn.cycle(&input)
        });
    }

    // #[bench]
    // fn cycle_64_131072_64(b: &mut Bencher) {
    //     let mut rng = Xoshiro256StarStar::seed_from_u64(1234);
    //     let mut nn = BinaryBrain::new(64, 64, 131200, &mut rng).unwrap();
    //     let input = [u8::MAX; 64];

    //     b.iter(|| {
    //         nn.cycle(&input)
    //     });
    // }
}