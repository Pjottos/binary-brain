use crate::*;
use crate::util::Xoshiro128PlusPlusAvx2;
// use rayon::prelude::*;
use rand::distributions::{Distribution, Uniform};
use std::cmp::Ordering;
use std::mem::transmute;

pub struct Genetic {
    population: Vec<(BinaryBrain, f64)>,
    p_mutate: u8,
    pop_select: Uniform<usize>,
    tournament_size: usize,
}

impl Genetic {
    pub fn new(initial: BinaryBrain, pop_size: usize, tournament_size: usize, mutation: u8) -> Result<Genetic> {
        if pop_size < 2 {
            return Err(BinaryBrainError::InvalidPopSize);
        }
        if tournament_size == 0 {
            return Err(BinaryBrainError::ZeroTournamentSize);
        }

        let mut pop = Vec::with_capacity(pop_size);
        pop.push((initial, f64::MIN));
        for _ in 0..pop_size - 1 {
            pop.push((BinaryBrain::from_template(&pop[0].0), f64::MIN));
        }

        Ok(Genetic {
            population: pop,
            p_mutate: mutation,
            pop_select: Uniform::new(0, pop_size),
            tournament_size: tournament_size,
        })
    }

    pub fn evaluate<F: FnMut(&mut BinaryBrain) -> f64>(&mut self, mut fitness: F) -> f64 {
        self.population.iter_mut().for_each(|p| {    
            p.1 = fitness(&mut p.0);
        });
        self.population.iter().max_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(Ordering::Equal)).unwrap().1
    }

    // TODO: somehow training fails to produce good results when using parallel evaluation when it works fine normally

    // pub fn evaluate_parallel<F: Fn(&mut BinaryBrain) -> f64 + Send + Sync>(&mut self, fitness: F) -> f64 {
    //     self.population.par_iter_mut().for_each(|p| {    
    //         p.1 = fitness(&mut p.0);
    //     });
    //     self.population.iter().max_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(Ordering::Equal)).unwrap().1
    // }

    pub fn breed(&mut self) {
        let mut new_pop = Vec::with_capacity(self.population.len());

        let weight_chunk_count = self.population[0].0.weights().len();
        let act_count = self.population[0].0.activations().len();
        let input_count = self.population[0].0.input_count();
        let output_count = self.population[0].0.output_count();

        let mut rng = thread_rng();
        let mut tmp = [0; 64];
        rng.fill_bytes(&mut tmp);
        let mut bulk_rng = Xoshiro128PlusPlusAvx2::new(tmp);

        let mut run_tournament = || {
            let mut fit = [f64::MIN, f64::MIN];
            let mut result = [&self.population[0].0, &self.population[0].0];
            for _ in 0..self.tournament_size {
                let idx = self.pop_select.sample(&mut rng);
                let candidate = &self.population[idx];
                for i in 0..2 {
                    if candidate.1 > fit[i] {
                        fit[i] = candidate.1;
                        result[i] = &candidate.0;
                        break;
                    }
                }
            }

            result
        };

        for _ in 0..self.population.len() / 2 {
            let parents = run_tournament();
            
            let mut weights = (Vec::with_capacity(weight_chunk_count), Vec::with_capacity(weight_chunk_count));
            let mut activations = (Vec::with_capacity(act_count), Vec::with_capacity(act_count));

            for i in (0..weight_chunk_count).step_by(4) {
                let mutations = bulk_rng.next_with_bias(self.p_mutate);
                let crossover = bulk_rng.next();
                
                for j in 0..4 {
                    let a = parents[0].weights()[i + j].0;
                    let b = parents[1].weights()[i + j].0;
                    
                    weights.0.push(NeuronChunk(((a & !crossover[j]) | (b & crossover[j])) ^ mutations[j]));
                    weights.1.push(NeuronChunk(((a & crossover[j]) | (b & !crossover[j])) ^ mutations[j]));
                }
            }

            for i in (0..act_count).step_by(32) {
                let mutate_triggers: [i8; 32] = unsafe { transmute(bulk_rng.next()) };
                let mutations: [i8; 32] = unsafe { transmute(bulk_rng.next()) };
                let crossover: [i8; 32] = unsafe { transmute(bulk_rng.next()) };

                for j in 0..32 {
                    if mutate_triggers[j] > 127 - self.p_mutate as i8 {
                        activations.0.push(Activation(mutations[j]));
                        activations.1.push(Activation(mutations[(j + 1) % 32]));
                    } else if crossover[j] > -1 {
                        activations.0.push(parents[1].activations()[i + j]);
                        activations.1.push(parents[0].activations()[i + j]);
                    } else {
                        activations.0.push(parents[0].activations()[i + j]);
                        activations.1.push(parents[1].activations()[i + j]);
                    }
                }
            }

            new_pop.push((BinaryBrain::with_parameters(weights.0, activations.0, input_count, output_count).unwrap(), f64::MIN));
            new_pop.push((BinaryBrain::with_parameters(weights.1, activations.1, input_count, output_count).unwrap(), f64::MIN));
        }

        self.population = new_pop;
    }
        
    pub fn clone_fittest(self) -> (BinaryBrain, f64) {
        let target = self.population.iter().max_by(
            |a, b| a.1.partial_cmp(&b.1).unwrap_or(Ordering::Equal)
        ).unwrap();

        (*target).clone()
    }
}


#[cfg(test)]
mod benches {
    use test::{Bencher};
    use super::*;

    // the genetic trainer has a linear time complexity with regard to popsize, so only test with popsize 1

    #[bench]
    fn genetic_breed_4096_16(b: &mut Bencher) {
        let brain = BinaryBrain::new(32, 32, 4096).unwrap();
        let mut trainer = Genetic::new(brain, 2, 16, 1).unwrap();
        let mut rng = thread_rng();
        let dist = Uniform::new(-128.0, 128.0);
        b.iter(|| {
            trainer.evaluate(|_| {
                dist.sample(&mut rng)
            });

            trainer.breed();
        });
    }

    #[bench]
    fn genetic_breed_512_16(b: &mut Bencher) {
        let brain = BinaryBrain::new(32, 32, 512).unwrap();
        let mut trainer = Genetic::new(brain, 2, 16, 1).unwrap();
        let mut rng = thread_rng();
        let dist = Uniform::new(-128.0, 128.0);
        b.iter(|| {
            trainer.evaluate(|_| {
                dist.sample(&mut rng)
            });

            trainer.breed();
        });
    }

    #[bench]
    fn genetic_breed_64_16(b: &mut Bencher) {
        let brain = BinaryBrain::new(32, 32, 64).unwrap();
        let mut trainer = Genetic::new(brain, 2, 16, 1).unwrap();
        let mut rng = thread_rng();
        let dist = Uniform::new(-128.0, 128.0);
        b.iter(|| {
            trainer.evaluate(|_| {
                dist.sample(&mut rng)
            });

            trainer.breed();
        });
    }
}