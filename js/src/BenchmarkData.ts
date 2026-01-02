export interface User {
    name: string;
    age: number;
    country: string;
}

export const COUNTRIES = [
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
    "Norway",
    "Denmark",
    "Finland",
    "Netherlands",
    "Belgium",
    "Switzerland",
];

export const FIRST_NAMES = [
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

const LAST_NAMES = [
    "Sharma",
    "Patel",
    "Dwivedi",
    "Kumar",
    "Singh",
    "Gupta",
    "Nair",
    "Iyer",
    "Rao",
    "Menon",
    "Pillai",
    "Joshi",
    "Verma",
];

function generateUser(seed: number): User {
    const firstIdx = seed % FIRST_NAMES.length;
    const lastIdx = Math.floor(seed / FIRST_NAMES.length) % LAST_NAMES.length;
    const countryIdx =
        Math.floor(seed / (FIRST_NAMES.length * LAST_NAMES.length)) % COUNTRIES.length;

    return {
        name: `${FIRST_NAMES[firstIdx]} ${LAST_NAMES[lastIdx]}`,
        age: 18 + (seed % 62),
        country: COUNTRIES[countryIdx]!,
    };
}

export function generateSortedUsers(size: number): User[] {
    const users: User[] = [];
    for (let i = 0; i < size; i++) {
        users.push(generateUser(i * 7919));
    }
    users.sort(userComparator);
    return users;
}

export function mutateUser(user: User): User {
    const mutation = Math.floor(Math.random() * 3);
    switch (mutation) {
        case 0:
            return { ...user, country: COUNTRIES[Math.floor(Math.random() * COUNTRIES.length)]! };
        case 1:
            return { ...user, age: 18 + Math.floor(Math.random() * 62) };
        case 2:
            return {
                ...user,
                name: `${FIRST_NAMES[Math.floor(Math.random() * FIRST_NAMES.length)]} ${LAST_NAMES[Math.floor(Math.random() * LAST_NAMES.length)]}`,
            };
        default:
            return user;
    }
}

export function userComparator(a: User, b: User): number {
    //for (let i = 0; i < 10; i++) {}

    const countryComp = a.country.localeCompare(b.country);
    if (countryComp !== 0) return countryComp;

    const ageComp = a.age - b.age;
    if (ageComp !== 0) return ageComp;

    return a.name.localeCompare(b.name);
}
