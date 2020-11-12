#![feature(test)]

extern crate test;

use std::iter::repeat_with;
use rand::prelude::*;

pub type Result<T> = std::result::Result<T, BinaryBrainError>;

#[derive(Debug)]
pub struct BinaryBrain {
    weight_matrix: Vec<u64>,
    values: Vec<u64>,
    act: Vec<u8>,
    input_count: usize,
    output_count: usize,
}

impl BinaryBrain {
    pub fn new(input_count: usize, output_count: usize, total_count: usize) -> Result<BinaryBrain> {
        if total_count % 64 != 0 {
            return Err(BinaryBrainError::TotalNotDivisbleBy64);
        }
        if input_count + output_count > total_count {
            return Err(BinaryBrainError::InputOutputAboveTotal);
        }

        let mut rng = thread_rng();
        let mut act = vec![0; total_count];
        rng.fill_bytes(act.as_mut_slice());

        let weight_count = total_count * total_count;

        Ok(BinaryBrain {
            weight_matrix: repeat_with(|| rng.next_u64()).take(weight_count / 64).collect(),
            values: vec![0; total_count / 64],
            act: act,
            input_count: input_count,
            output_count: output_count,
        })
    }

    #[inline]
    pub fn cycle(&mut self, input: &[u8], output: &mut Vec<bool>) -> Result<()> {
        if input.len() != self.input_count {
            return Err(BinaryBrainError::WrongInputShape);
        }
        
        output.clear();
        output.reserve(self.output_count);
        
        let neuron_count = self.act.len();
        for i in (0..neuron_count).step_by(64) {
            // write the values of connections in batches to reduce memory i/o
            let mut batch: u64 = 0;

            // construct the batch by evaluating 64 neurons
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
                if j >= output_start {
                    // we iterate the neurons sequentially so this works
                    output.push(fire);
                }
            }
            
            self.values[i / 64] = batch;
        }

        Ok(())
    }

    #[inline]
    unsafe fn calc_sum(&self, neuron: usize) -> i32 {
        let mut sum = 0;
        let weight_row_size = self.values.len();
        let row_start = neuron * weight_row_size;

        // for every row in the weight matrix (it's square)
        for i in 0..weight_row_size {
            // perform an xnor with the weights and values, then add the popcount of that value
            // and subtract the zero count
            // this produces the following value per connection/weight:
            //          !weight   weight  
            // !value |    1    |  -1    |
            // -------+---------+--------+
            // value  |   -1    |   1    |

            // couldn't get the bounds check optimized away unfortunately
            // not having the bounds check allows the compiler to vectorize the loop
            let weighted = !(self.weight_matrix.get_unchecked(row_start + i) ^ self.values.get_unchecked(i));
            sum += weighted.count_ones() as i32;
            sum -= weighted.count_zeros() as i32;
        }

        sum
    }
}

#[derive(Debug)]
pub enum BinaryBrainError {
    TotalNotDivisbleBy64,
    InputOutputAboveTotal,
    WrongInputShape,
}


#[cfg(test)]
mod benches {
    use test::{Bencher};
    use crate::BinaryBrain;

    #[bench]
    fn cycle_64_64_64(b: &mut Bencher) {
        let mut nn = BinaryBrain::new(64, 64, 192).unwrap();
        let input = [u8::MAX; 64];
        let mut output = vec![];

        b.iter(|| {
            nn.cycle(&input, &mut output).unwrap();
        });
    } 

    #[bench]
    fn cycle_64_512_64(b: &mut Bencher) {
        let mut nn = BinaryBrain::new(64, 64, 640).unwrap();
        let input = [u8::MAX; 64];
        let mut output = vec![];

        b.iter(|| {
            nn.cycle(&input, &mut output).unwrap();
        });
    }

    #[bench]
    fn cycle_64_4096_64(b: &mut Bencher) {
        let mut nn = BinaryBrain::new(64, 64, 4224).unwrap();
        let input = [u8::MAX; 64];
        let mut output = vec![];

        b.iter(|| {
            nn.cycle(&input, &mut output).unwrap();
        });
    }

    #[bench]
    fn cycle_64_32768_64(b: &mut Bencher) {
        let mut nn = BinaryBrain::new(64, 64, 32896).unwrap();
        let input = [u8::MAX; 64];
        let mut output = vec![];

        b.iter(|| {
            nn.cycle(&input, &mut output).unwrap();
        });
    }
}