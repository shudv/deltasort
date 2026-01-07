use rand::Rng;
use std::cell::Cell;

// ============================================================================
// BENCHMARK DATA
// ============================================================================

const COUNTRIES: &[&str] = &[
    "USA",
    "Canada",
    "UK",
    "Germany",
    "France",
    "Spain",
    "Italy",
    "Japan",
    "Australia",
    "Brazil",
    "India",
    "China",
    "Mexico",
    "Argentina",
    "Sweden",
];

const FIRST_NAMES: &[&str] = &[
    "Vijay",
    "Meera",
    "Akash",
    "Kashish",
    "Sunita",
    "Aviral",
    "Saumya",
    "Aman",
    "Sanjay",
    "Kavitha",
    "Radhika",
    "Meenakshi",
    "Suresh",
    "Krishna",
];

const LAST_NAMES: &[&str] = &[
    "Sharma", "Patel", "Dwivedi", "Kumar", "Singh", "Gupta", "Nair", "Iyer", "Rao", "Menon",
    "Pillai", "Joshi", "Verma",
];

#[derive(Clone, Debug, Default)]
pub struct User {
    name: String,
    age: u32,
    country: String,
}

impl User {
    pub fn generate(seed: usize) -> Self {
        let first_idx = seed % FIRST_NAMES.len();
        let last_idx = (seed / FIRST_NAMES.len()) % LAST_NAMES.len();
        let country_idx = (seed / (FIRST_NAMES.len() * LAST_NAMES.len())) % COUNTRIES.len();

        User {
            name: format!("{} {}", FIRST_NAMES[first_idx], LAST_NAMES[last_idx]),
            age: 18 + (seed % 62) as u32,
            country: COUNTRIES[country_idx].to_string(),
        }
    }

    pub fn mutate(&mut self, rng: &mut impl Rng) {
        let field = rng.gen_range(0..3);
        match field {
            0 => {
                let first = FIRST_NAMES[rng.gen_range(0..FIRST_NAMES.len())];
                let last = LAST_NAMES[rng.gen_range(0..LAST_NAMES.len())];
                self.name = format!("{} {}", first, last);
            }
            1 => self.age = 18 + rng.gen_range(0..62),
            _ => self.country = COUNTRIES[rng.gen_range(0..COUNTRIES.len())].to_string(),
        }
    }
}

pub fn user_comparator(a: &User, b: &User) -> std::cmp::Ordering {
    a.country
        .cmp(&b.country)
        .then_with(|| a.age.cmp(&b.age))
        .then_with(|| a.name.cmp(&b.name))
}

thread_local! {
    static COMPARISON_COUNT: Cell<u64> = const { Cell::new(0) };
}

pub fn counting_comparator(a: &User, b: &User) -> std::cmp::Ordering {
    COMPARISON_COUNT.with(|c| c.set(c.get() + 1));
    user_comparator(a, b)
}

pub fn reset_comparison_count() {
    COMPARISON_COUNT.with(|c| c.set(0));
}

pub fn get_comparison_count() -> u64 {
    COMPARISON_COUNT.with(|c| c.get())
}

pub fn generate_sorted_users(n: usize) -> Vec<User> {
    let mut users: Vec<User> = (0..n).map(User::generate).collect();
    users.sort_by(user_comparator);
    users
}

pub fn sample_distinct_indices(rng: &mut impl Rng, n: usize, k: usize) -> Vec<usize> {
    let mut arr: Vec<usize> = (0..n).collect();
    for i in 0..k {
        let j = rng.gen_range(i..n);
        arr.swap(i, j);
    }
    arr.truncate(k);
    arr
}
