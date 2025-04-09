
#[derive(Debug, Clone, Copy)]
pub struct DataStep<'a, T: Copy> {
    now: usize,
    all: usize,
    step: usize,
    data: &'a Vec<T>,
    // start: usize,
}

impl<'a, T: Copy> DataStep<'a, T> {
    pub fn new(data: &'a Vec<T>, step: usize, now: usize) -> Self {
        let all = data.len();
        Self { now, all, step, data }
    }

    // pub fn reset(&mut self) {
    //     self.now = self.start;
    // }
}

impl<T: Copy> Iterator for DataStep<'_, T> {
    type Item = T;
    fn next(&mut self) -> Option<Self::Item> {
        if self.now > self.all {
            return None;
        }
        let res = self.data[self.now];
        self.now += self.step;
        Some(res)
    }
}
