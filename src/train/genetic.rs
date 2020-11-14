use crate::*;
// use rayon::prelude::*;
use rand::distributions::{Distribution, Uniform};
use rand_xoshiro::Xoshiro256PlusPlus;
use std::cmp::Ordering;
use std::iter;

pub struct Genetic {
    population: Vec<(BinaryBrain, f64)>,
    p_mutate: usize,
    parent_count: usize,
    weight_crossovers: usize,
    act_crossovers: usize,
    pop_select: Uniform<usize>,
    tournament_size: usize,
    rng: Xoshiro256PlusPlus,
}

impl Genetic {
    pub fn new(initial: BinaryBrain, pop_size: usize, tournament_size: usize, mutation: usize, parent_count: usize, weight_crossovers: usize, act_crossovers: usize) -> Result<Genetic> {
        if pop_size == 0 {
            return Err(BinaryBrainError::ZeroPopSize);
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
            parent_count: parent_count,
            weight_crossovers: weight_crossovers,
            act_crossovers: act_crossovers,
            pop_select: Uniform::new(0, pop_size),
            tournament_size: tournament_size,
            rng: Xoshiro256PlusPlus::from_entropy(),
        })
    }

    pub fn evaluate<F: FnMut(&mut BinaryBrain) -> f64 + Send + Sync>(&mut self, mut fitness: F) -> f64 {
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
        let pop_count = self.population.len();
        let weight_count = self.population[0].0.weights().len() * 64;
        let act_count = self.population[0].0.activations().len();
        
        // needed because we cannot mutate self while borrowing the brains
        let mut rng = self.rng.clone();
        self.rng.jump();

        let weight_crossover_dist = Uniform::new(0, weight_count);
        let act_crossover_dist = Uniform::new(0, act_count);

        self.population = iter::repeat_with(|| {
            let parents: Vec<&BinaryBrain> = iter::repeat_with(|| {
                iter::repeat_with(|| {
                    let idx = self.pop_select.sample(&mut rng);
                    (idx, self.population[idx].1)
                })
                    .take(self.tournament_size)
                    .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(Ordering::Equal))
                    .map(|x| &self.population[x.0].0)
                    .unwrap()
            }).take(self.parent_count).collect();

            let mut current_parent = 0;
            let mut crossovers: Vec<usize> = weight_crossover_dist.sample_iter(&mut rng)
                .take(self.weight_crossovers)
                .collect();
            crossovers.sort_unstable();
            crossovers.dedup();
            crossovers.reverse();
            let mut next_crossover = crossovers.pop().unwrap_or(usize::MAX);

            let weights = (0..weight_count / 64).map(|i| {
                let mut seg = parents[current_parent].weights()[i];

                if i * 64 >= next_crossover {
                    let idx = next_crossover as u32 % 64;
                    // clear the bits [idx, 64)
                    let mask = u64::MAX.overflowing_shr(64 - idx).0;
                    seg &= mask;

                    // get the corresponding bits from the next parent
                    current_parent = (current_parent + 1) % self.parent_count;
                    let seg2 = parents[current_parent].weights()[i];
                    seg |= seg2 & !mask;

                    next_crossover = crossovers.pop().unwrap_or(usize::MAX);
                }

                if self.p_mutate != 0 {
                    let mut mutate = rng.next_u64();
                    for _ in 1..self.p_mutate {
                        mutate &= rng.next_u64();
                    }
                    seg ^= mutate;
                }

                seg
            }).collect();

            crossovers = act_crossover_dist.sample_iter(&mut rng)
                .take(self.act_crossovers)
                .collect();
            crossovers.sort_unstable();
            crossovers.dedup();
            crossovers.reverse();

            next_crossover = crossovers.pop().unwrap_or(usize::MAX);

            let activations = (0..act_count).map(|i| { 
                if i == next_crossover {
                    current_parent = (current_parent + 1) % self.parent_count;
                    next_crossover = crossovers.pop().unwrap_or(usize::MAX);
                }

                let mut act = parents[current_parent].activations()[i];
                
                if self.p_mutate != 0 {
                    // check p_mutate random bits and if any of them is set, apply a mutation.
                    // this corresponds to a probability of 0.5^p_mutate
                    let random = rng.next_u64();
                    let r = random >> (64 - self.p_mutate);
                    if r.count_ones() == 0 {
                        // use a (most likely) different part of the previously generated random number
                        // this is to save a call to rng
                        act = random as u8;
                    }
                }

                act
            }).collect();
            
            (BinaryBrain::with_parameters(weights, activations, parents[0].input_count(), parents[0].output_count()).unwrap(), f64::MIN)
        }).take(pop_count).collect();
    }

    pub fn clone_fittest(&self) -> (BinaryBrain, f64) {
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
    fn genetic_breed_4096_8_7_2_32768_256(b: &mut Bencher) {
        let brain = BinaryBrain::new(32, 32, 4096).unwrap();
        let mut trainer = Genetic::new(brain, 1, 4096, 8, 7, 32768, 256).unwrap();
        let mut rng = Xoshiro256PlusPlus::from_entropy();
        let dist = Uniform::new(-128.0, 128.0);
        b.iter(|| {
            trainer.evaluate(|_| {
                dist.sample(&mut rng)
            });

            trainer.breed();
        });
    }

    #[bench]
    fn genetic_breed_4096_8_0_2_1048576_4096(b: &mut Bencher) {
        let brain = BinaryBrain::new(32, 32, 4096).unwrap();
        let mut trainer = Genetic::new(brain, 1, 8, 0, 2, 32768, 256).unwrap();
        let mut rng = Xoshiro256PlusPlus::from_entropy();
        let dist = Uniform::new(-128.0, 128.0);
        b.iter(|| {
            trainer.evaluate(|_| {
                dist.sample(&mut rng)
            });

            trainer.breed();
        });
    }

    #[bench]
    fn genetic_breed_4096_8_32_2_0_0(b: &mut Bencher) {
        let brain = BinaryBrain::new(32, 32, 4096).unwrap();
        let mut trainer = Genetic::new(brain, 1, 8, 32, 2, 0, 0).unwrap();
        let mut rng = Xoshiro256PlusPlus::from_entropy();
        let dist = Uniform::new(-128.0, 128.0);
        b.iter(|| {
            trainer.evaluate(|_| {
                dist.sample(&mut rng)
            });

            trainer.breed();
        });
    }
}