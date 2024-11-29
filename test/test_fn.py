import unittest
from operator import add
from functools import partial
from src import fn

class Test_Maybe(unittest.TestCase):
    def test_maybe_nothing(self):
        res = self.safe_divide(10, 2).flatmap(partial(self.safe_divide, b=0))
        self.assertTrue(res.is_empty())

    def test_maybe_just(self):
        res = self.safe_divide(8, 2).flatmap(partial(self.safe_divide, b=2))
        self.assertFalse(res.is_empty())
        self.assertEqual(res.value, 2)

    def test_apply(self):
        cadd = fn.curry(add)

        res1 = fn.Just(cadd).apply(fn.Just(3)).apply(fn.Just(7))
        self.assertFalse(res1.is_empty())
        self.assertEqual(res1.value, 10)

        res2 = fn.Just(cadd).apply(fn.Nothing).apply(fn.Just(7))
        self.assertTrue(res2.is_empty())

        res3 = fn.Just(cadd).apply(fn.Just(3)).apply(fn.Nothing)
        self.assertTrue(res3.is_empty())

        res4 = fn.Nothing.apply(fn.Just(3)).apply(fn.Nothing)
        self.assertTrue(res4.is_empty())

    def test_get_or_else(self):
        self.assertEqual(fn.Just(5).get_or_else(10), 5)
        self.assertEqual(fn.Nothing.get_or_else(10), 10)

    def safe_divide(self, a: float, b: float) -> fn.Maybe[float]:
        if b == 0:
            return fn.Nothing
        return fn.Just(a // b)


class Test_Either(unittest.TestCase):
    def test_either_left(self):
        res = self.safe_divide(10, 2).flatmap(partial(self.safe_divide, b=0))
        self.assertTrue(res.is_left())
        self.assertEqual(res.value, "Division by zero")

    def test_either_right(self):
        res = self.safe_divide(8, 2).flatmap(partial(self.safe_divide, b=2))
        self.assertTrue(res.is_right())
        self.assertEqual(res.value, 2)

    def test_map(self):
        res = fn.Left(10).map(lambda x: x + 10)
        self.assertTrue(res.is_left())
        self.assertEqual(res.value, 10)

    def test_apply(self):
        cadd = fn.curry(add)
        res1 = fn.Right(cadd).apply(fn.Right(3)).apply(fn.Right(7))
        self.assertTrue(res1.is_right())
        self.assertEqual(res1.value, 10)

        res2 = fn.Right(cadd).apply(fn.Left(3)).apply(fn.Right(7))
        self.assertTrue(res2.is_left())
        self.assertEqual(res2.value, 3)

        res3 = fn.Right(cadd).apply(fn.Right(3)).apply(fn.Left(7))
        self.assertTrue(res3.is_left())
        self.assertEqual(res3.value, 7)

        res4 = fn.Left(cadd).apply(fn.Right(3)).apply(fn.Right(7))
        self.assertTrue(res4.is_left())
        self.assertEqual(res4.value, cadd)

    def test_get_or_else(self):
        self.assertEqual(fn.Right(5).get_or_else(10), 5)
        self.assertEqual(fn.Left(5).get_or_else(10), 10)

    def safe_divide(self, a: float, b: float) -> fn.Either[str, float]:
        if b == 0:
            return fn.Left("Division by zero")
        return fn.Right(a / b)


class TestFunctions(unittest.TestCase):
    def test_curry2(self):
        cadd = fn.curry(add)
        cadd10 = cadd(10)
        self.assertEqual(cadd10(2), 12)

    def test_curry3(self):
        def add3(x, y, z):
            return x + y + z

        cadd = fn.curry(add3)
        cadd10 = cadd(10)
        cadd10Then20 = cadd10(20)
        self.assertEqual(cadd10Then20(2), 32)

    def test_liftA2(self):
        cadd = fn.curry(add)
        self.assertEqual(fn.liftA2(cadd, fn.Just(10), fn.Just(2)), fn.Just(12))
        self.assertEqual(fn.liftA2(cadd, fn.Nothing, fn.Just(2)), fn.Nothing)
        self.assertEqual(fn.liftA2(cadd, fn.Just(10), fn.Nothing), fn.Nothing)

        self.assertEqual(fn.liftA2(cadd, fn.Right(10), fn.Right(2)), fn.Right(12))
        self.assertEqual(fn.liftA2(cadd, fn.Left(10), fn.Right(2)), fn.Left(10))
        self.assertEqual(fn.liftA2(cadd, fn.Right(10), fn.Left(2)), fn.Left(2))


if __name__ == "__main__":
    unittest.main()
