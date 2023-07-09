use std::collections::HashMap;
use std::error::Error;

pub fn is_prime(p: u32) -> bool {
    for i in 2..=(p / 2) {
        if p % i == 0 {
            return false;
        }
    }
    true
}

pub struct FpUnitsDiscLogSolver {
    pub prime: u32,
}

impl FpUnitsDiscLogSolver {
    pub fn new(p: u32) -> Self {
        assert!(is_prime(p));
        FpUnitsDiscLogSolver { prime: p }
    }

    pub fn fast_power(&self, mut g: u32, mut exp: u32) -> u32 {
        assert!(g % self.prime != 0);
        let mut result = 1;
        // let base = g;
        while exp > 0 {
            if exp % 2 == 1 {
                result *= g;
                result %= self.prime;
            }
            g *= g;
            g %= self.prime;
            exp /= 2;
        }
        result
    }

    pub fn compute_inverse(&self, g: u32) -> u32 {
        assert!(g % self.prime != 0);
        self.fast_power(g, self.prime - 2)
    }

    pub fn compute_order(&self, g: u32) -> u32 {
        assert!(g % self.prime != 0);
        let mut g = g;
        let base = g;
        let mut n: u32 = 1;
        while g != 1 {
            g *= base;
            g %= self.prime;
            n += 1;
        }
        n
    }

    pub fn shanks_algorithm(&self, base: u32, num: u32) -> Option<u32> {
        assert!(base % self.prime != 0 && num % self.prime != 0);
        // First compute order of base
        let order = self.compute_order(base);
        let n = f32::floor(f32::sqrt(order as f32)) as u32 + 1;

        // Generate our first list
        let mut list1 = HashMap::new();
        let mut g = 1;

        for i in 0..(n as i32) {
            list1.insert(g, i);
            g *= base;
            g %= self.prime;
        }

        // Insert final value for base and exponent into first list
        list1.insert(g, n as i32);

        let mut u = 1;
        let base_inverse = self.compute_inverse(g);

        for i in 0..=(n as i32) {
            // Check if we have a match
            if let Some(x) = list1.get(&((num * u) % self.prime)) {
                let res = x + (i * (n as i32));
                return Some(res as u32);
            }
            // Otherwise update value of u
            u *= base_inverse;
            u %= self.prime;
        }
        None
    }

    pub fn shanks_algorithm_with_output(
        &self,
        base: u32,
        num: u32,
    ) -> (Option<u32>, HashMap<u32, i32>, HashMap<u32, i32>) {
        assert!(base % self.prime != 0 && num % self.prime != 0);

        // Compute order of base
        let order = self.compute_order(base);
        let n = f32::floor(f32::sqrt(order as f32)) as u32 + 1;

        // Generate our first list
        let mut list1 = HashMap::new();
        let mut g = 1;

        for i in 0..(n as i32) {
            list1.insert(g, i);
            g *= base;
            g %= self.prime;
        }

        // insert final value into the list1
        list1.insert(g, n as i32);

        // Now compute the inverse of base^n, note g = base^n from our previous computation
        let mut list2 = HashMap::new();
        let mut u = 1;
        let base_inverse = self.compute_inverse(g);

        for i in 0..=(n as i32) {
            // First always insert into list2
            list2.insert(num * u, i);
            // Then check if we have a match in list1
            if let Some(x) = list1.get(&((num * u) % self.prime)) {
                let res = x + (i * (n as i32));
                return (Some(res as u32), list1, list2);
            }

            // otherwise update u and proceed
            u *= base_inverse;
            u %= self.prime;
        }

        // there is no discrete logarithm for the given base and num
        (None, list1, list2)
    }

    // pub fn shanks_algorithm_with_lists
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_compute_inverse() {
        let solver = FpUnitsDiscLogSolver::new(71);
        let base = 11;
        let inverse = solver.compute_inverse(11);

        println!("g = {}, g inverse = {}", base, inverse);
        println!("g * g^-1 = {}", (base * inverse) % 71);

        assert_eq!((base * inverse) % 71, 1);

        let solver = FpUnitsDiscLogSolver::new(17389);
        let base = 9704;
        let inverse = solver.compute_inverse(9704);

        println!("g = {}, g inverse = {}", base, inverse);
        println!("g * g^-1 = {}", (base * inverse) % 17389);

        assert_eq!((base * inverse) % 17389, 1);
    }

    #[test]
    fn test_compute_order() {
        let solver = FpUnitsDiscLogSolver::new(71);
        let base = 11;
        let order: u32 = solver.compute_order(base);
        println!("g = {}, order of g = {}", base, order);
        println!("g^n = {}", solver.fast_power(base, order));

        assert_eq!(solver.fast_power(base, order), 1);

        let solver = FpUnitsDiscLogSolver::new(17389);
        let base = 9704;
        let order: u32 = solver.compute_order(base);
        println!("g = {}, order of g = {}", base, order);
        println!("g^n = {}", solver.fast_power(base, order));

        assert_eq!(solver.fast_power(base, order), 1);
    }

    #[test]
    fn test_shanks_algorithm() {
        let solver = FpUnitsDiscLogSolver::new(71);
        let base = 11;
        if let Some(x) = solver.shanks_algorithm(base, 23) {
            println!("{}^{} = {}", base, x, solver.fast_power(base, x));
            assert_eq!(solver.fast_power(base, x), 23);
        } else {
            panic!();
        }

        let solver = FpUnitsDiscLogSolver::new(17389);
        let base = 9704;
        if let Some(x) = solver.shanks_algorithm(base, 13896) {
            println!("{}^{} = {}", base, x, solver.fast_power(base, x));
            assert_eq!(solver.fast_power(base, x), 13896);
        }

        assert!(true);
    }

    #[test]
    fn test_shanks_algorithm_with_output() {
        let solver = FpUnitsDiscLogSolver::new(71);
        let base = 11;
        match solver.shanks_algorithm_with_output(base, 23) {
            (Some(x), l1, l2) => {
                println!("list1: {:#?}", l1);
                println!("list2: {:#?}", l2);
                println!("log = {}", x);
                println!("{}^{} = {}", base, x, solver.fast_power(base, x));

                assert_eq!(solver.fast_power(base, x), 23);
            }
            (None, _l1, _l2) => panic!(),
            _ => panic!(),
        }

        let solver = FpUnitsDiscLogSolver::new(17389);
        let base = 9704;

        match solver.shanks_algorithm_with_output(base, 13896) {
            (Some(x), l1, l2) => {
                println!("list1: {:#?}", l1);
                println!("list2: {:#?}", l2);
                println!("log = {}", x);
                println!("{}^{} = {}", base, x, solver.fast_power(base, x));

                assert_eq!(solver.fast_power(base, x), 13896);
            }
            (None, _l1, _l2) => panic!(),
            _ => panic!(),
        }

        assert!(true);
    }
}
