use actix_web::{body::BoxBody, http::header::ContentType, HttpResponse, Responder};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::error::Error;
use std::marker::{Send, Sync};
use std::thread;

/// Simple function that uses a brute force technique to check if `p` is prime or not
/// Returns true if `p` is prime, otherwise it returns false
pub fn is_prime(p: u32) -> bool {
    for i in 2..=(p / 2) {
        if p % i == 0 {
            return false;
        }
    }
    true
}

/// Function that uses the seives algorithm for computing all primes up to and including a given input `n`.
pub fn compute_primes(n: u32) -> Vec<u32> {
    let mut primes = vec![true; (n + 1) as usize];
    let mut curr = 2_u32;

    while curr <= n {
        if primes[curr as usize] {
            let multiples = (curr * 2..=curr * (n / curr)).step_by(curr as usize);
            for multiple in multiples {
                primes[multiple as usize] = false;
            }
        }
        curr += 1;
    }

    primes
        .into_iter()
        .enumerate()
        .filter(|(i, b)| *i != 0 && *i != 1 && *b)
        .map(|(i, _)| i as u32)
        .collect()
}

/// Simple function that uses brute force to attempt a factorization of `n`
/// returns a vector of (prime, exponent) where prime is the prime factor and exponent is the highest power of that prime that divides n
pub fn factor(mut n: u32) -> Vec<(u32, u32, u32)> {
    let potential_prime_factors = PrimeGenerator::new(n / 2);
    let mut result = vec![];
    for factor in potential_prime_factors {
        if factor > n {
            break;
        } else if n % factor == 0 {
            let mut exp = 0;
            while n % factor == 0 && n > 0 {
                n /= factor;
                exp += 1;
            }
            result.push((factor, exp, u32::pow(factor, exp)));
        }
    }
    result
}

/// Function that implements the fast powering algorithm. Takes `prime` a prime, `g` a base and `exp` an exponent that we would
/// like to raise `g` to the power of. The return value is `g` raised to `exp` modulo `prime`, the function panics if `prime` is not prime.
pub fn fast_power_fp(prime: u32, g: u32, exp: u32) -> u32 {
    assert!(is_prime(prime));
    fast_power_fp_unchecked(prime, g, exp)
}

/// Function that implements the fast powering algorithm. Takes `prime` a prime, `g` a base and `exp` an exponent that we would
/// like to raise `g` to the power of. The return value is `g` raised to `exp` modulo `prime`,
/// the function is unchecked, so it will not panic if `prime` is not actually a prime.
pub fn fast_power_fp_unchecked(prime: u32, mut g: u32, mut exp: u32) -> u32 {
    let mut result = 1;
    while exp > 0 {
        if exp % 2 == 1 {
            result *= g;
            result %= prime;
        }

        g *= g;
        g %= prime;
        exp /= 2;
    }
    result
}

/// Funciton that implements the fast power algorithm for any `g` raised to the power `e` with modulus `m`.
pub fn fast_power(m: u32, mut g: u32, mut exp: u32) -> u32 {
    if g == 0 {
        return 0;
    }
    let mut result = 1;

    while exp > 0 {
        if exp % 2 == 1 {
            result *= g;
            result %= m;
        }

        g *= g;
        g %= m;
        exp /= 2;
    }

    result
}

/// Simple function to compute the greatest common divisor of a pair of integers.
/// The method panices if both of `a` and `b` are equal to zero, since in this case the greatest common divisor in undefined.
pub fn gcd(mut a: u32, mut b: u32) -> u32 {
    assert!(!(a == 0 && b == 0));
    gcd_unchecked(a, b)
}

/// Simple function to compute the greatest common divisor of a pair of integers.
/// The funciton does not panic if both `a` == 0 and `b` == 0 since it is unchecked.
pub fn gcd_unchecked(mut a: u32, mut b: u32) -> u32 {
    let mut r = a % b;
    while r > 0 {
        a = b;
        b = r;
        r = a % b;
    }
    b
}

/// Function that will compute the multiplicative inverse of a given integer `g` modulo `prime` where `prime` must be a prime.
/// The function will panic if `prime` is not prime or if `prime` divides `g`.
pub fn compute_inverse_fp(prime: u32, g: u32) -> u32 {
    assert!(is_prime(prime) && g % prime != 0);
    fast_power_fp_unchecked(prime, g, prime - 2)
}

/// Function that will compute the multiplicative inverse of a given integer `g` modulo `prime` where `prime` must be a prime.
/// The function does not check whether `prime` is actually a prime or not.
pub fn compute_inverse_fp_unchecked(prime: u32, g: u32) -> u32 {
    fast_power_fp_unchecked(prime, g, prime - 2)
}

/// Function that will compute the inverse of `a` modulo `b`. Function returns an Option<u32>, the some variant if the inverse exists otherwise it returns None.
pub fn compute_inverse_mod_n(a: u32, b: u32) -> Option<u32> {
    // Table to build up answer
    let mut table = vec![(0, 1), (1, 0)];
    let mut curr_a = a;
    let mut curr_b = b;

    let mut q = curr_a / curr_b;
    let mut r = curr_a % curr_b;
    let mut i = 1;

    while r > 0 {
        let (u1, u2) = (table[i - 1].0, table[i].0);
        let (v1, v2) = (table[i - 1].1, table[i].1);
        let curr_u = u2 * q + u1;
        let curr_v = v2 * q + v1;
        table.push((curr_u, curr_v));
        curr_a = curr_b;
        curr_b = r;
        q = curr_a / curr_b;
        r = curr_a % curr_b;
        i += 1;
    }
    // Check all possible solutions, if curr_b == 1 then the algorithm must have computed
    //a valid solution that we can use to compute the inverse of a modulo b
    if curr_b == 1 {
        let (mut u, mut v) = table[table.len() - 1];

        if let (Some(x), Some(y)) = (a.checked_mul(u), b.checked_mul(v)) {
            match x.checked_add(y) {
                Some(res) if res == 1 => return Some(u % b),
                _ => {}
            }
            match x.checked_sub(y) {
                Some(res) if res == 1 => return Some(u % b),
                _ => {}
            }
            match y.checked_sub(x) {
                Some(res) if res == 1 => {
                    u %= b;
                    return Some(b - u);
                }
                _ => {}
            }
        };

        if let (Some(x), Some(y)) = (a.checked_mul(v), b.checked_mul(u)) {
            match x.checked_add(y) {
                Some(res) if res == 1 => return Some(v % b),
                _ => {}
            }
            match x.checked_sub(y) {
                Some(res) if res == 1 => return Some(v % b),
                _ => {}
            }
            match y.checked_sub(x) {
                Some(res) if res == 1 => {
                    v %= b;
                    return Some(b - v);
                }
                _ => {}
            }
        };
    }

    None
}

/// Computes the order of `m1` in the group of units from the ring quotient ring Z/(`m2`).
pub fn compute_order(m1: u32, m2: u32) -> u32 {
    assert!(gcd(m1, m2) == 1);
    compute_order_unchecked(m1, m2)
}

/// Computes the order of `m1` in the group of units from the ring quotient ring Z/(`m2`).
/// The funciton is unchecked and therefore does not bother to check if `m1` and `m2` are relatively prime.
pub fn compute_order_unchecked(mut m1: u32, m2: u32) -> u32 {
    let base = m1;
    let mut n = 1;
    while m1 != 1 {
        m1 *= base;
        m1 %= m2;
        n += 1;
    }
    n
}

/// Function that will compute the order of an integer `g` in the group of units from the field Fp, where p is the prime `prime`.
/// The funciton will panic if `prime` is not prime or `prime` divides `g`.
pub fn compute_order_fp(prime: u32, g: u32) -> u32 {
    assert!(is_prime(prime) && g % prime != 0);
    compute_order_fp_unchecked(prime, g)
}

/// Function that will compute the order of an integer `g` in the group of units from the field Fp, where p is the prime `prime`.
/// The funciton will not panic if `prime` is not prime or `prime` divides `g`, since the function is unchecked. Useful when one already knows that `prime` is prime.
pub fn compute_order_fp_unchecked(prime: u32, mut g: u32) -> u32 {
    let base = g;
    let mut n = 1;
    while g != 1 {
        g *= base;
        g %= prime;
        n += 1;
    }
    n
}

/// Function will solve the discrete logarithm of `num` with base `g` in the group of units from the field Fp, where p is the prime supplied argument `prime`.
/// The funciton will panic if `prime` is not prime, `prime` divides `num` or `prime` divides `g`.
pub fn shanks_algorithm(prime: u32, g: u32, num: u32) -> Option<u32> {
    assert!(is_prime(prime) && g % prime != 0 && num % prime != 0);
    shanks_algorithm_unchecked(prime, g, num)
}

/// An unchecked version of `shanks_algorithm` useful and more efficient when the caller has already established the necessary conditions for the algorithm to be successful.
/// Function will solve the discrete logarithm of `num` with base `g` in the group of units from the field Fp, where p is the prime supplied argument `prime`.
/// The funciton will not panic if `prime` is not prime, `prime` divides `num` or `prime` divides `g` since it is the unchecked version.
pub fn shanks_algorithm_unchecked(prime: u32, g: u32, num: u32) -> Option<u32> {
    // First compute order of base
    let base_order = compute_order_fp_unchecked(prime, g);

    let n = f32::floor(f32::sqrt(base_order as f32)) as u32 + 1;

    let mut prod = 1;
    // first list for generating collisions
    let mut list1 = HashMap::new();

    for i in 0..(n as i32) {
        list1.insert(prod, i);
        prod *= g;
        prod %= prime;
    }

    // insert final value of first list
    list1.insert(prod, n as i32);

    // now compute second list, we don't actually need to allocate any memory for this only set up variables
    let mut u = 1;
    let prod_inverse = compute_inverse_fp_unchecked(prime, prod);

    for i in 0..=(n as i32) {
        // check if we have a collision
        if let Some(x) = list1.get(&(num * u % prime)) {
            let res = (x + (i * (n as i32))) as u32;
            return Some(res);
        }

        // otherwise update the value of u
        u *= prod_inverse;
        u %= prime;
    }

    None
}

/// Function will solve the discrete logarithm of `num` with base `g` in the group of units from the field Fp, where p is the prime supplied argument `prime`.
/// The function will also return the lists tat were generated the collision. The funciton will panic if `prime` is not prime,
/// `prime` divides `num` or `prime` divides `g`.
pub fn shanks_algorithm_with_output(
    prime: u32,
    g: u32,
    num: u32,
) -> (Option<u32>, u32, u32, HashMap<u32, i32>, HashMap<u32, i32>) {
    assert!(is_prime(prime) && g % prime != 0 && num % prime != 0);
    shanks_algorithm_with_output_unchecked(prime, g, num)
}

/// Function will solve the discrete logarithm of `num` with base `g` in the group of units from the field Fp, where p is the prime supplied argument `prime`.
/// The function will also return the lists tat were generated the collision. The funciton will panic if `prime` is not prime,
/// `prime` divides `num` or `prime` divides `g`.
pub fn shanks_algorithm_with_output_unchecked(
    prime: u32,
    g: u32,
    num: u32,
) -> (Option<u32>, u32, u32, HashMap<u32, i32>, HashMap<u32, i32>) {
    // Compute order of base
    let order = compute_order(prime, g);

    let n = f32::floor(f32::sqrt(order as f32)) as u32 + 1;

    // Generate our first list
    let mut list1 = HashMap::new();
    let mut prod = 1;

    for i in 0..(n as i32) {
        list1.insert(prod, i);
        prod *= g;
        prod %= prime;
    }

    // insert final value into the list1
    list1.insert(g, n as i32);

    // Now compute the inverse of base^n, note g = base^n from our previous computation
    let mut list2 = HashMap::new();
    let mut u = 1;
    let base_inverse = compute_inverse_fp(prime, prod);

    for i in 0..=(n as i32) {
        // First always insert into list2
        list2.insert(num * u, i);
        // Then check if we have a match in list1
        if let Some(x) = list1.get(&((num * u) % prime)) {
            let res = x + (i * (n as i32));
            return (Some(res as u32), base_inverse, n, list1, list2);
        }

        // otherwise update u and proceed
        u *= base_inverse;
        u %= prime;
    }

    // there is no discrete logarithm for the given base and num
    (None, base_inverse, n, list1, list2)
}

/// A function that will solve a simultaneous system of congruences using the chinese remainder theorem.
/// Returns an Option<u32>. The return value is some if a solutione exists, otherwise it will return None
pub fn solve_congruences(congruences: Vec<(u32, u32)>) -> Option<u32> {
    let (mut x, mut m) = (congruences[0].0, congruences[0].1);
    for i in 1..congruences.len() {
        let (a_i, m_i) = congruences[i];
        let y = if let Some(m_inv) = compute_inverse_mod_n(m, m_i) {
            let temp_x = x % m_i;
            if temp_x > a_i {
                ((a_i + (m_i - temp_x)) * m_inv) % m_i
            } else {
                ((a_i - temp_x) * m_inv) % m_i
            }
        } else {
            // Otherwise no solution exists for this system of congruences
            return None;
        };
        let x_ = ((m * y) % (m * m_i)) + (x % (m * m_i));
        x = x_;
        m *= m_i;
    }

    Some(x % m)
}

/// A function that does essentially the same thing as `solve_congruences` except that it returns the output of
/// all intermediate congruences as computed via chinese remainder theorem, and returns the solution
pub fn solve_congruences_with_output(
    congruences: Vec<(u32, u32)>,
) -> Option<(u32, Vec<(u32, u32, u32, u32, u32)>)> {
    let (mut x, mut m) = (congruences[0].0, congruences[0].1);
    let mut congruence_solutions = vec![(x, m, 1, x, m)];

    for i in 1..congruences.len() {
        let (a_i, m_i) = congruences[i];
        let y = if let Some(m_inv) = compute_inverse_mod_n(m, m_i) {
            let temp_x = x % m_i;
            if temp_x > a_i {
                ((a_i + (m_i - temp_x)) * m_inv) % m_i
            } else {
                ((a_i - temp_x) * m_inv) % m_i
            }
        } else {
            // Otherwise no solution exists for this system of congruences
            return None;
        };

        let x_ = (m * y) % (m * m_i) + (x % (m * m_i));
        x = x_;
        m *= m_i;
        congruence_solutions.push((x, m, y, a_i, m_i));
    }

    Some((x, congruence_solutions))
}

/// A function that will solve the discrete logarithm using shanks algorithm for prime powers in an efficient way.
/// The funciton is unchecked for performance reasons, so assumes that all sufficient conditions on the inputs have been met, the output is not correct otherwise.
pub fn solve_prime_power_unchecked(
    prime_modulus: u32,
    base: u32,
    num: u32,
    prime_factor: u32,
    power: u32,
) -> Option<u32> {
    let mut x = 0;
    let mut u = 1;
    let mut v = u32::pow(prime_factor, power - 1);
    // let b = u32::pow(base, u32::pow(prime, v));
    let b = fast_power_fp_unchecked(prime_modulus, base, v);

    // println!("{b}");

    for _i in 0..power {
        let curr_inverse = compute_inverse_fp_unchecked(
            prime_modulus,
            fast_power_fp_unchecked(prime_modulus, base, x),
        );
        let curr_num =
            fast_power_fp_unchecked(prime_modulus, (num * curr_inverse) % prime_modulus, v);

        if let Some(x_i) = shanks_algorithm_unchecked(
            prime_modulus,
            b,
            fast_power_fp_unchecked(prime_modulus, (num * curr_inverse) % prime_modulus, v),
        ) {
            x += x_i * u;
            x %= u32::pow(prime_factor, power);
            u *= prime_factor;
            v /= prime_factor;
        } else {
            return None;
        }
    }

    Some(x)
}

/// A method that essentially performs the same algorithm as `solve_prime_power_unchecked` however it collects
/// the generated data that may be of interest and returns an `Option<PrimeSolver>` as its output instead of the single answer that represents a solution.
pub fn solve_prime_power_with_output_unchecked(
    prime_modulus: u32,
    base: u32,
    num: u32,
    prime_factor: u32,
    power: u32,
) -> Option<SolvePrimePowerOutput> {
    let mut temp_solutions = HashMap::new();

    let mut x = 0;
    let mut u = 1;
    let mut v = u32::pow(prime_factor, power - 1);

    let temp_base = fast_power_fp_unchecked(prime_modulus, base, v);

    for i in 0..power {
        let curr_inverse = compute_inverse_fp_unchecked(
            prime_modulus,
            fast_power_fp_unchecked(prime_modulus, base, x),
        );
        let curr_num =
            fast_power_fp_unchecked(prime_modulus, (num * curr_inverse) % prime_modulus, v);

        if let (Some(x_i), base_inverse, n, list1, list2) = shanks_algorithm_with_output_unchecked(
            prime_modulus,
            temp_base,
            fast_power_fp_unchecked(prime_modulus, (num * curr_inverse) % prime_modulus, v),
        ) {
            temp_solutions.insert(
                (x_i, i, curr_inverse),
                ShanksOutput {
                    prime: prime_factor,
                    base: temp_base,
                    num: curr_num,
                    n,
                    log: x_i,
                    base_inverse,
                    collision_list1: list1,
                    collision_list2: list2,
                },
            );
            x += x_i * u;
            x %= u32::pow(prime_factor, power);
            u *= prime_factor;
            v /= prime_factor;
        } else {
            return None;
        }
    }

    Some(SolvePrimePowerOutput {
        prime: prime_factor,
        exp: power,
        num,
        base,
        temp_base,
        temp_solutions,
        solution: x,
    })
}

/// A struct that will generate prime numbers that will generate prime numbers up to and including some final number.
pub struct PrimeGenerator {
    n: u32,
    primes: Vec<bool>,
    curr: u32,
}

impl PrimeGenerator {
    /// Creates a new `PrimeGenerator`
    pub fn new(n: u32) -> Self {
        let primes = vec![true; (n + 1) as usize];
        let curr = 2_u32;
        PrimeGenerator { n, primes, curr }
    }
}

impl std::iter::Iterator for PrimeGenerator {
    type Item = u32;

    fn next(&mut self) -> Option<Self::Item> {
        while self.curr <= self.n {
            if self.primes[self.curr as usize] {
                let multiples =
                    (self.curr * 2..=self.curr * (self.n / self.curr)).step_by(self.curr as usize);
                for multiple in multiples {
                    self.primes[multiple as usize] = false;
                }
                let res = self.curr;
                self.curr += 1;
                return Some(res);
            } else {
                self.curr += 1;
            }
        }
        None
    }
}

/// A struct for collecting all relevant data when solving the discrete logarithm using shanks algorithm
#[derive(Serialize, Debug)]
pub struct ShanksOutput {
    pub prime: u32,
    pub base: u32,
    pub num: u32,
    pub n: u32,
    pub log: u32,
    pub base_inverse: u32,
    pub collision_list1: HashMap<u32, i32>,
    pub collision_list2: HashMap<u32, i32>,
}

impl Responder for ShanksOutput {
    type Body = BoxBody;

    fn respond_to(self, req: &actix_web::HttpRequest) -> HttpResponse<Self::Body> {
        let body = serde_json::to_string(&self).unwrap();
        HttpResponse::Ok()
            .content_type(ContentType::json())
            .body(body)
    }
}

impl std::fmt::Display for ShanksOutput {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f, 
            "prime: {}\n base: {}\n num: {}\n n: {}\n log: {}\n base_inverse: {}\n list1: {:?}\n list2: {:?}\n", 
            self.prime, self.base, self.num, self.n, self.log, self.base_inverse, self.collision_list1, self.collision_list2
        )
    }
}

/// A struct for holding all of the relevant data when solving a prime power using the efficient shanks algorithm
#[derive(Serialize, Debug)]
pub struct SolvePrimePowerOutput {
    pub prime: u32,
    pub exp: u32,
    pub num: u32,
    pub base: u32,
    pub temp_base: u32,
    pub temp_solutions: HashMap<(u32, u32, u32), ShanksOutput>,
    pub solution: u32,
}

impl std::fmt::Display for SolvePrimePowerOutput {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "prime: {}\n exp: {}\n num: {}\n base: {}\n temp_base: {}\n temp_solutions: {:?}\n solution: {}",
            self.prime, self.exp, self.num, self.base, self.temp_base, self.temp_solutions, self.solution,
        )
    }
}

/// A struct for holding the output of a pollhig-hellman algorithm result.
 #[derive(Serialize, Debug)]
pub struct PollhigHellmanOutput {
    prime_modulus: u32,
    base: u32,
    num: u32,
    order: u32,
    order_prime_factorization: Vec<(u32, u32, u32)>,
    prime_power_solution_output: Vec<SolvePrimePowerOutput>,
    congruences: Vec<(u32, u32, u32, u32, u32)>,
    solution: u32,
}

impl std::fmt::Display for PollhigHellmanOutput {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "prime_modulus: {}\n base: {}\n num: {}\n order: {}\n order_factorization: {:?}\n prime_power_solution_output: {:#?}\n congruences: {:?}\n solution: {}",
            self.prime_modulus, self.base, self.num, self.order, self.order_prime_factorization, self.prime_power_solution_output, self.congruences, self.solution
        )
    }
}

/// A struct that encapsulates all the necessary functions for solving the discrete logarithm in the group of units from the field Fp.
/// A `FpUnitsDiscLogSolver` may be preferable in many cases since only one check is required at initialization to ensure the modulus is prime,
/// where as, many repeated checks that the modulus is prime will be needed if using only the functions declared in this module
#[derive(Clone, Copy)]
pub struct FpUnitsDiscLogSolver {
    pub prime: u32,
}

impl FpUnitsDiscLogSolver {
    /// Create a new `FpUnitsDiscLogSolver`, will panic if `p` is not prime.
    pub fn new(p: u32) -> Self {
        assert!(is_prime(p));
        FpUnitsDiscLogSolver { prime: p }
    }

    /// Performs the fast power algorithm for computing a base raised to an exponenet in the group of units from Fp.
    /// In this method the parameter `g` is the base and `exp` is the power we wish to raise `g` to.
    /// The method panics if `g` % self.prime == 0, since in this case `g` is not in the group of units to begin with.
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

    /// Computes the inverse of a given element `g` from the group of units in Fp.
    /// Method will panic if `g` % self.prime == 0, since in this case `g` is not in the group of units to begin with.
    pub fn compute_inverse(&self, g: u32) -> u32 {
        assert!(g % self.prime != 0);
        self.fast_power(g, self.prime - 2)
    }

    /// Computes the order of a given element `g` from the group of units in Fp.
    /// Method will panic if `g` % self.prime == 0, since in this case `g` is not in the group of units to begin with.
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

    /// Attempts to solve the discrete logarithm problem. `num` is the value whose logarithm we are trying to find with the given `base`.
    /// If the discrete logarithm exists, a `Some(u32)` will be returned, otherwise `None` will be returned.
    /// Method will panic if `base` % self.prime == 0 or `num` % self.prime == 0, since in this case either `base` or `num` is not in the group of units to begin with.
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

    /// Attempts to solve the discrete logarithm problem. `num` is the value whose logarithm we are trying to find with the given `base`.
    /// This method also returns the data that may be of interest to some that was generated during the algorithm in an attempt to find the logarithm, eg the list of elements that were generated when trying to find a collision, etc...
    /// If the discrete logarithm exists, a `Some(u32)` will be returned, otherwise `None` will be returned in the tuple containing all of the outputs.
    /// Method will panic if `base` % self.prime == 0 or `num` % self.prime == 0, since in this case either `base` or `num` is not in the group of units to begin with.
    pub fn shanks_algorithm_with_output(
        &self,
        base: u32,
        num: u32,
    ) -> (Option<u32>, u32, u32, HashMap<u32, i32>, HashMap<u32, i32>) {
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
            list2.insert(num * u % self.prime, i);
            // Then check if we have a match in list1
            if let Some(x) = list1.get(&((num * u) % self.prime)) {
                let res = x + (i * (n as i32));
                return (Some(res as u32), u, n, list1, list2);
            }

            // otherwise update u and proceed
            u *= base_inverse;
            u %= self.prime;
        }

        // there is no discrete logarithm for the given base and num
        (None, u, n, list1, list2)
    }

    /// A method that will solve the discrete logarithm where the modulus is a prime power.
    /// Can be called as a stand alone function or as a helper method.
    pub fn solve_prime_power(&self, base: u32, num: u32, prime: u32, exp: u32) -> Option<u32> {
        let mut x = 0;
        let mut u = 1;
        let mut v = u32::pow(prime, exp - 1);
        // let b = u32::pow(base, u32::pow(prime, v));
        let b = self.fast_power(base, v);

        for _i in 0..exp {
            let curr_inverse = self.compute_inverse(self.fast_power(base, x));
            if let Some(x_i) =
                self.shanks_algorithm(b, self.fast_power((num * curr_inverse) % self.prime, v))
            {
                x += x_i * u;
                x %= u32::pow(prime, exp);
                u *= prime;
                v /= prime;
            } else {
                return None;
            }
        }

        Some(x)
    }

    /// A method for computing the discrete logarithm using Pollhig-Hellman algorithm.
    /// Method will panic if `base` % self.prime == 0 or `num` % self.prime == 0, since in this case either `base` or `num` is not in the group of units to begin with.
    pub fn pollhig_hellman(&self, base: u32, num: u32) -> Option<u32> {
        assert!(base % self.prime != 0 && num % self.prime != 0);
        // First compute order of base
        let order = self.compute_order(base);

        // Compute prime powers of order
        let prime_powers = factor(order);
        let mut prime_pow_solution_congruences = vec![];

        for (prime, power, prime_power) in prime_powers {
            // Build the system of congruences, where each modulus is a prime power factor of the order of the given base `g`
            if let Some(y) = self.solve_prime_power(
                self.fast_power(base, order / prime_power),
                self.fast_power(num, order / prime_power),
                prime,
                power,
            ) {
                prime_pow_solution_congruences.push((y, prime_power));
            } else {
                // Has no solution
                return None;
            }
        }

        // Use chinese remainder theorm to solve the simulataneous system of congruents
        match solve_congruences(prime_pow_solution_congruences) {
            Some(sol) => Some(sol % self.prime),
            _ => None,
        }
    }

    /// A method for computing the discrete logarithm using Pollhig-Hellman algorithm, however,
    /// it employs concurrency when solving the discrete logarithm problems for single prime power factors,
    /// this is for performance reasons, but may not actually be necessary for many use cases.
    /// Method will panic if `base` % self.prime == 0 or `num` % self.prime == 0, since in this case either `base` or `num` is not in the group of units to begin with.
    pub fn pollhig_hellman_concurrent(&self, base: u32, num: u32) -> Option<u32> {
        assert!(base % self.prime != 0 && num % self.prime != 0);
        // First compute order of base
        let order = self.compute_order(base);

        // Compute prime power factors of order
        let prime_powers = factor(order);

        // Collect join handles from spawning the threads
        let mut solution_handles = vec![];

        for (prime, power, prime_power) in prime_powers {
            let prime_modulus = self.prime;
            // Spawn a thread for each prime power factor of order for solving the current discrete logarithm
            solution_handles.push(thread::spawn(move || -> Result<(u32, u32), String> {
                if let Some(y) = solve_prime_power_unchecked(
                    prime_modulus,
                    fast_power_fp_unchecked(prime_modulus, base, order / prime_power),
                    fast_power_fp_unchecked(prime_modulus, num, order / prime_power),
                    prime,
                    power,
                ) {
                    Ok((y, prime_power))
                } else {
                    Err(String::from("no solution"))
                }
            }));
        }

        // convert join handles into vector of congruences
        if let Ok(congruences) = solution_handles
            .into_iter()
            .map(|handle| handle.join().unwrap())
            .collect::<Result<Vec<(u32, u32)>, String>>()
        {
            if let Some(x) = solve_congruences(congruences) {
                return Some(x);
            }
        }

        None
    }

    /// A method for computing the discrete logarithm using Pollhig-Hellman algorithm, however it returns the generated data that is of interest from the procedure.
    /// The method employs concurrency when solving the discrete logarithm problems for single prime power factors,
    /// this is for performance reasons, but may not actually be necessary for many use cases.
    /// Method will panic if `base` % self.prime == 0 or `num` % self.prime == 0, since in this case either `base` or `num` is not in the group of units to begin with.
    pub fn pollhig_hellman_with_output(&self, base: u32, num: u32) -> Option<PollhigHellmanOutput> {
        assert!(base % self.prime != 0 && num % self.prime != 0);
        // First compute order of base
        let order = self.compute_order(base);

        // Compute prime power factors of order
        let prime_powers = factor(order);

        // Collect join handles from spawning the threads
        let mut solution_handles = vec![];

        for (prime, power, prime_power) in &prime_powers {
            // Move out of referenced values so we can pass them to a new thread
            let prime_modulus = self.prime;
            let (prime, power, prime_power) = (*prime, *power, *prime_power);
            // Spawn a thread for each prime power factor of order for solving the current discrete logarithm
            solution_handles.push(thread::spawn(
                move || -> Result<(SolvePrimePowerOutput, u32), String> {
                    if let Some(out) = solve_prime_power_with_output_unchecked(
                        prime_modulus,
                        fast_power_fp_unchecked(prime_modulus, base, order / prime_power),
                        fast_power_fp_unchecked(prime_modulus, num, order / prime_power),
                        prime,
                        power,
                    ) {
                        Ok((out, prime_power))
                    } else {
                        Err(String::from("no solution"))
                    }
                },
            ));
        }

        // Now for each join handle, solve the congruences and save the output. In addition save the prime power solver output in a vector as well
        let mut prime_power_outputs = vec![];
        let mut congruences = vec![];

        for handle in solution_handles {
            if let Ok((res, prime_power)) = handle.join().unwrap() {
                congruences.push((res.solution, prime_power));
                prime_power_outputs.push(res);
            } else {
                return None;
            }
        }

        // Now solve the congruences, saving the output of each sub-congruence, to illustrate the algorithm
        if let Some((solution, congruence_output)) = solve_congruences_with_output(congruences) {
            Some(PollhigHellmanOutput {
                prime_modulus: self.prime,
                base: base,
                num: num,
                order: order,
                order_prime_factorization: prime_powers,
                prime_power_solution_output: prime_power_outputs,
                congruences: congruence_output,
                solution,
            })
        } else {
            None
        }
    }
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
            (Some(x), _, _, l1, l2) => {
                println!("list1: {:#?}", l1);
                println!("list2: {:#?}", l2);
                println!("log = {}", x);
                println!("{}^{} = {}", base, x, solver.fast_power(base, x));

                assert_eq!(solver.fast_power(base, x), 23);
            }
            (None, _, _, _l1, _l2) => panic!(),
            _ => panic!(),
        }

        let solver = FpUnitsDiscLogSolver::new(17389);
        let base = 9704;

        match solver.shanks_algorithm_with_output(base, 13896) {
            (Some(x), _, _, l1, l2) => {
                println!("list1: {:#?}", l1);
                println!("list2: {:#?}", l2);
                println!("log = {}", x);
                println!("{}^{} = {}", base, x, solver.fast_power(base, x));

                assert_eq!(solver.fast_power(base, x), 13896);
            }
            (None, _, _, _l1, _l2) => panic!(),
            _ => panic!(),
        }

        assert!(true);
    }

    #[test]
    fn test_compute_primes() {
        let prime = 31_u32;
        assert_eq!(
            vec![2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31],
            compute_primes(prime)
        );
        assert_eq!(
            vec![2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31],
            compute_primes(32)
        );

        println!("{:?}", compute_primes(32));
        println!("{:?}", compute_primes(1000));

        // tests passed
        assert!(true);
    }

    #[test]
    fn test_prime_generator() {
        let prime_generator = PrimeGenerator::new(1000);

        assert_eq!(prime_generator.collect::<Vec<u32>>(), compute_primes(1000));
    }

    #[test]
    fn test_factor() {
        let N = 2 * 2 * 28 * 3 * 7 * 15 * 9;
        println!("{N}");
        let factors = factor(N as u32);
        println!("{:?}", factors);

        assert_eq!(vec![(2, 4, 16), (3, 4, 81), (5, 1, 5), (7, 2, 49)], factors);
    }

    #[test]
    fn test_solve_prime_power() {
        let solver = FpUnitsDiscLogSolver::new(11251);
        if let Some(x) = solver.solve_prime_power(5448, 6909, 5, 4) {
            println!("{x}");
            assert!(x % 11251 == 511);
        } else {
            println!("panicing returned None");
            panic!();
        }
    }

    #[test]
    fn test_compute_inverse_mod_n() {
        let a = 73_u32;
        let b = 25_u32;

        if let Some(u) = compute_inverse_mod_n(a, b) {
            println!("{u}");
        } else {
            // panic!();
        }

        let a = 67;
        let b = 71;

        if let Some(u) = compute_inverse_mod_n(a, b) {
            println!("{u}");
        }

        let a = 101;
        let b = 746497 * 71;

        if let Some(u) = compute_inverse_mod_n(a, b) {
            println!("{u}");
        }

        if let Some(u) = compute_inverse_mod_n(b, a) {
            println!("{u}");
        }

        assert!(true);
    }

    #[test]
    fn test_solve_congruences() {
        let congruences = vec![(2, 3), (3, 5), (2, 7)];
        if let Some(solution) = solve_congruences(congruences.clone()) {
            println!("{:?}", solution);
            assert!(congruences.iter().all(|(a, m)| solution % m == *a));
        }

        let congruences = vec![(2, 3), (3, 7), (4, 16)];
        if let Some(solution) = solve_congruences(congruences.clone()) {
            println!("{:?}", solution);
            assert!(congruences.iter().all(|(a, m)| solution % m == *a));
        }

        let congruences = vec![(37, 43), (22, 49), (18, 71)];
        if let Some(solution) = solve_congruences(congruences.clone()) {
            println!("{:?}", solution);
            assert!(congruences.iter().all(|(a, m)| solution % m == *a));
        }

        let congruences = vec![(37, 43), (22, 49), (18, 71), (93, 101), (75, 97)];
        if let Some(solution) = solve_congruences(congruences.clone()) {
            println!("{:?}", solution);
            assert!(congruences.iter().all(|(a, m)| solution % m == *a));
        }
    }

    #[test]
    fn test_pollhig_hellman() {
        let prime = 11251;
        let g = 23;
        let h = 9689;
        let solver = FpUnitsDiscLogSolver::new(prime);
        if let Some(x) = solver.pollhig_hellman(g, h) {
            println!("{x}");
            println!(
                "{} ^ {x} = {} mod {}",
                g,
                solver.fast_power(g, x),
                solver.prime
            );

            assert_eq!(solver.fast_power(g, x), h);
        }

        let prime = 71;
        let g = 11;
        let h = 21;
        let solver = FpUnitsDiscLogSolver::new(prime);
        if let Some(x) = solver.pollhig_hellman(g, h) {
            println!("{x}");
            println!(
                "{} ^ {x} = {} mod {}",
                g,
                solver.fast_power(g, x),
                solver.prime
            );

            assert_eq!(solver.fast_power(g, x), h);
        }

        let prime = 593;
        let g = 156;
        let h = 116;
        let solver = FpUnitsDiscLogSolver::new(prime);
        if let Some(x) = solver.pollhig_hellman(g, h) {
            println!("{x}");
            println!(
                "{} ^ {x} = {} mod {}",
                g,
                solver.fast_power(g, x),
                solver.prime
            );

            assert_eq!(solver.fast_power(g, x), h);
        }

        let prime = 3571;
        let g = 650;
        let h = 2213;
        let solver = FpUnitsDiscLogSolver::new(prime);
        if let Some(x) = solver.pollhig_hellman(g, h) {
            println!("{x}");
            println!(
                "{} ^ {x} = {} mod {}",
                g,
                solver.fast_power(g, x),
                solver.prime
            );

            assert_eq!(solver.fast_power(g, x), h);
        }
    }

    #[test]
    fn test_pollhig_hellman_concurrent() {
        let prime = 11251;
        let g = 23;
        let h = 9689;
        let solver = FpUnitsDiscLogSolver::new(prime);
        if let Some(x) = solver.pollhig_hellman_concurrent(g, h) {
            println!(
                "{} ^ {x} = {} mod {}",
                g,
                solver.fast_power(g, x),
                solver.prime
            );

            assert_eq!(solver.fast_power(g, x), h);
        } else {
            panic!("did not successfully compute result");
        }

        let prime = 71;
        let g = 11;
        let h = 21;
        let solver = FpUnitsDiscLogSolver::new(prime);
        if let Some(x) = solver.pollhig_hellman_concurrent(g, h) {
            println!("{x}");
            println!(
                "{} ^ {x} = {} mod {}",
                g,
                solver.fast_power(g, x),
                solver.prime
            );

            assert_eq!(solver.fast_power(g, x), h);
        } else {
            panic!("did not successfully compute result");
        }

        let prime = 593;
        let g = 156;
        let h = 116;
        let solver = FpUnitsDiscLogSolver::new(prime);
        if let Some(x) = solver.pollhig_hellman_concurrent(g, h) {
            println!("{x}");
            println!(
                "{} ^ {x} = {} mod {}",
                g,
                solver.fast_power(g, x),
                solver.prime
            );

            assert_eq!(solver.fast_power(g, x), h);
        } else {
            panic!("did not successfully compute result");
        }

        let prime = 3571;
        let g = 650;
        let h = 2213;
        let solver = FpUnitsDiscLogSolver::new(prime);
        if let Some(x) = solver.pollhig_hellman_concurrent(g, h) {
            println!("{x}");
            println!(
                "{} ^ {x} = {} mod {}",
                g,
                solver.fast_power(g, x),
                solver.prime
            );

            assert_eq!(solver.fast_power(g, x), h);
        } else {
            panic!("did not successfully compute result");
        }
    }

    #[test]
    fn test_solve_prime_power_unchecked() {
        let prime = 11251;
        let base = 5448;
        let num = 6909;
        let prime_factor = 5;
        let prime_power = 4;
        if let Some(x) = solve_prime_power_unchecked(prime, base, num, prime_factor, prime_power) {
            println!("{x}");
        }
    }

    #[test]
    fn test_shanks_algorithm_unchecked() {
        let base = 11089;
        let num = 6320;
        let prime = 11251;

        if let Some(x) = shanks_algorithm_unchecked(prime, base, num) {
            println!("{x}");
        }
    }

    #[test]
    fn test_pollhig_hellman_with_output() {
        let prime = 11251;
        let g = 23;
        let h = 9689;
        let solver = FpUnitsDiscLogSolver::new(prime);
        if let Some(out) = solver.pollhig_hellman_with_output(g, h) {
            println!(
                "{} ^ {} = {} mod {}",
                g,
                out.solution,
                solver.fast_power(g, out.solution),
                solver.prime
            );

            println!("{}", out);

            assert_eq!(solver.fast_power(g, out.solution), h);
        } else {
            panic!("did not successfully compute result");
        }

        let prime = 71;
        let g = 11;
        let h = 21;
        let solver = FpUnitsDiscLogSolver::new(prime);
        if let Some(x) = solver.pollhig_hellman_concurrent(g, h) {
            println!("{x}");
            println!(
                "{} ^ {x} = {} mod {}",
                g,
                solver.fast_power(g, x),
                solver.prime
            );

            assert_eq!(solver.fast_power(g, x), h);
        } else {
            panic!("did not successfully compute result");
        }

        let prime = 593;
        let g = 156;
        let h = 116;
        let solver = FpUnitsDiscLogSolver::new(prime);
        if let Some(x) = solver.pollhig_hellman_concurrent(g, h) {
            println!("{x}");
            println!(
                "{} ^ {x} = {} mod {}",
                g,
                solver.fast_power(g, x),
                solver.prime
            );

            assert_eq!(solver.fast_power(g, x), h);
        } else {
            panic!("did not successfully compute result");
        }

        let prime = 3571;
        let g = 650;
        let h = 2213;
        let solver = FpUnitsDiscLogSolver::new(prime);
        if let Some(x) = solver.pollhig_hellman_concurrent(g, h) {
            println!("{x}");
            println!(
                "{} ^ {x} = {} mod {}",
                g,
                solver.fast_power(g, x),
                solver.prime
            );

            assert_eq!(solver.fast_power(g, x), h);
        } else {
            panic!("did not successfully compute result");
        }
    }
}
