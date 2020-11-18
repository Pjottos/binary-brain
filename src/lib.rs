#![feature(test)]

extern crate test;

use rand::prelude::*;
use byteorder::{LittleEndian, ReadBytesExt, WriteBytesExt};
use std::iter::repeat_with;
use std::fs;
use std::io;
use std::path::Path;
use std::mem::size_of;


pub type Result<T> = std::result::Result<T, BinaryBrainError>;

pub mod train;
mod util;

#[derive(Debug, Clone)]
pub struct BinaryBrain {
    weight_matrix: Vec<NeuronChunk>,
    values: Vec<NeuronChunk>,
    act: Vec<Activation>,
    input_count: usize,
    output_count: usize,
    neuron_count: usize,
}

impl BinaryBrain {
    pub fn new(input_count: usize, output_count: usize, total_count: usize) -> Result<BinaryBrain> {
        if total_count % (size_of::<NeuronChunk>() * 8) != 0 {
            return Err(BinaryBrainError::TotalNotDivisbleByChunkSize);
        }
        if input_count + output_count > total_count {
            return Err(BinaryBrainError::InputOutputAboveTotal);
        }

        let mut rng = thread_rng();
        let mut act = vec![Activation::default(); total_count];
        for i in 0..act.len() {
            act[i].0 = rng.gen();
        }

        let weight_count = total_count * total_count;

        Ok(BinaryBrain {
            weight_matrix: repeat_with(|| NeuronChunk(rng.gen()) ).take(weight_count / (size_of::<NeuronChunk>() * 8)).collect(),
            values: vec![NeuronChunk::default(); total_count / (size_of::<NeuronChunk>() * 8)],
            act: act,
            input_count: input_count,
            output_count: output_count,
            neuron_count: total_count,
        })
    }

    pub fn from_template(template: &BinaryBrain) -> BinaryBrain {
        Self::new(
            template.input_count,
            template.output_count,
            template.act.len()
        ).unwrap()
    }

    pub fn with_parameters(weight_matrix: Vec<NeuronChunk>, activations: Vec<Activation>, input_count: usize, output_count: usize) -> Result<BinaryBrain> {
        let total_count = activations.len();
        if input_count + output_count > total_count {
            return Err(BinaryBrainError::InputOutputAboveTotal);
        }
        if weight_matrix.len() != (total_count * total_count) / (size_of::<NeuronChunk>() * 8) {
            return Err(BinaryBrainError::InvalidWeightActivationCombo);
        }

        Ok(BinaryBrain {
            weight_matrix: weight_matrix,
            values: vec![NeuronChunk::default(); total_count / (size_of::<NeuronChunk>() * 8)],
            act: activations,
            input_count: input_count,
            output_count: output_count,
            neuron_count: total_count,
        })
    }

    pub fn from_file<P: AsRef<Path>>(path: P) -> io::Result<BinaryBrain> {
        let mut file = fs::File::open(path)?;

        let input_count = file.read_u64::<LittleEndian>()? as usize;
        let output_count = file.read_u64::<LittleEndian>()? as usize;
        let total_count = file.read_u64::<LittleEndian>()? as usize;

        let weight_chunks = (total_count * total_count) / (size_of::<NeuronChunk>() * 8);
        let mut weights = Vec::with_capacity(weight_chunks);
        for _ in 0..weight_chunks {
            let chunk = file.read_u64::<LittleEndian>()?;
            weights.push(NeuronChunk(chunk));
        }

        let mut act = Vec::with_capacity(total_count);
        for _ in 0..total_count {
            let chunk = file.read_i8()?;
            act.push(Activation(chunk));
        }

        Ok(BinaryBrain {
            weight_matrix: weights,
            values: vec![NeuronChunk::default(); total_count / (size_of::<NeuronChunk>() * 8)],
            act: act,
            input_count: input_count,
            output_count: output_count,
            neuron_count: total_count,
        })
    }

    pub fn write_to_file<P: AsRef<Path>>(&self, target: P) -> io::Result<()> {
        let mut file = fs::OpenOptions::new()
            .write(true)
            .truncate(true)
            .create(true)
            .open(target)?;
        
        file.write_u64::<LittleEndian>(self.input_count as u64)?;
        file.write_u64::<LittleEndian>(self.output_count as u64)?;
        file.write_u64::<LittleEndian>(self.neuron_count as u64)?;

        for chunk in self.weight_matrix.iter() {
            file.write_u64::<LittleEndian>(chunk.0)?;
        }

        for chunk in self.act.iter() {
            file.write_i8(chunk.0)?;
        }
        
        Ok(())
    }

    #[inline]
    pub fn cycle(&mut self, input: &[Activation], output: &mut Vec<(bool, i32)>) -> Result<()> {
        if input.len() != self.input_count {
            return Err(BinaryBrainError::WrongInputShape);
        }
        
        output.clear();
        output.reserve(self.output_count); 
        let output_start = self.act.len() - self.output_count;

        for (i, v) in self.act.iter().enumerate() {
            let chunk_size = size_of::<NeuronChunk>() * 8;
            let mut sum = self.calc_sum(i);
            
            if i < self.input_count {
                sum += input[i].0 as i32;
            }

            let fire = sum > v.0 as i32;

            if fire {
                self.values[i / chunk_size].0 |= 1 << (i % chunk_size);
            } else {
                self.values[i / chunk_size].0 &= !(1 << (i % chunk_size));
            }

            if i >= output_start {
                output.push((fire, sum));
            }
        }

        Ok(())
    }

    #[inline]
    fn calc_sum(&self, neuron: usize) -> i32 {
        let mut sum = 0;
        let row_size = self.values.len();

        let weight_iter = self.weight_matrix[neuron * row_size..(neuron + 1) * row_size].iter();
        let value_iter = self.values[0..row_size].iter();
        for (weights, values) in weight_iter.zip(value_iter) {
            let weighted = !(weights.0 ^ values.0);
            let popcount = weighted.count_ones() as i32; 
            sum += popcount;
            sum -= (size_of::<NeuronChunk>() * 8) as i32 - popcount;
        }

        sum
    }

    #[inline]
    pub fn weights(&self) -> &[NeuronChunk] {
        self.weight_matrix.as_slice()
    }

    #[inline]
    pub fn activations(&self) -> &[Activation] {
        self.act.as_slice()
    }

    #[inline]
    pub fn input_count(&self) -> usize {
        self.input_count
    }

    #[inline]
    pub fn output_count(&self) -> usize {
        self.output_count
    }
}

#[repr(transparent)]
#[derive(Debug, Copy, Clone, Default)]
pub struct NeuronChunk(pub u64);

#[repr(transparent)]
#[derive(Debug, Copy, Clone, Default)]
pub struct Activation(pub i8);

#[derive(Debug)]
pub enum BinaryBrainError {
    TotalNotDivisbleByChunkSize,
    InputOutputAboveTotal,
    WrongInputShape,
    InvalidPopSize,
    ZeroTournamentSize,
    InvalidWeightActivationCombo,
}


#[cfg(test)]
mod benches {
    use test::{Bencher, black_box};
    use crate::*;

    #[bench]
    fn cycle_64(b: &mut Bencher) {
        let mut nn = BinaryBrain::new(32, 32, 64).unwrap();
        let mut output = vec![];

        b.iter(|| {
            let input = black_box(&[Activation(0); 32]);
            nn.cycle(input, &mut output).unwrap();
        });
    }

    #[bench]
    fn cycle_512(b: &mut Bencher) {
        let mut nn = BinaryBrain::new(32, 32, 512).unwrap();
        let mut output = vec![];

        b.iter(|| {
            let input = black_box(&[Activation(0); 32]);
            nn.cycle(input, &mut output).unwrap();
        });
    } 

    #[bench]
    fn cycle_4096(b: &mut Bencher) {
        let mut nn = BinaryBrain::new(32, 32, 4096).unwrap();
        let mut output = vec![];

        b.iter(|| {
            let input = black_box(&[Activation(0); 32]);
            nn.cycle(input, &mut output).unwrap();
        });
    }

    #[bench]
    fn cycle_32768(b: &mut Bencher) {
        let mut nn = BinaryBrain::new(32, 32, 32768).unwrap();
        let mut output = vec![];

        b.iter(|| {
            let input = black_box(&[Activation(0); 32]);
            nn.cycle(input, &mut output).unwrap();
        });
    }
}