#pragma once

#include <vamp/vector.hh>

namespace vamp::rng
{
    template <std::size_t dim>
    struct Halton
    {
        static constexpr const std::array<float, 32> primes{
            3.F,
            5.F,
            7.F,
            11.F,
            13.F,
            17.F,
            19.F,
            23.F,
            29.F,
            31.F,
            37.F,
            41.F,
            43.F,
            47.F,
            53.F,
            59.F,
            61.F,
            67.F,
            71.F,
            73.F,
            79.F,
            83.F,
            89.F,
            97.F,
            101.F,
            103.F,
            107.F,
            109.F,
            113.F,
            127.F,
            131.F,
            137.F};

        explicit Halton(FloatVector<dim> b_in, std::size_t skip_iterations = 0) noexcept : b(b_in)
        {
            initialize(skip_iterations);
        }

        Halton(std::initializer_list<FloatT> v, std::size_t skip_iterations = 0) noexcept
          : Halton(FloatVector<dim>::pack_and_pad(v), skip_iterations)
        {
        }

        explicit Halton(std::size_t skip_iterations = 0)
          : Halton(
                []()
                {
                    alignas(FloatVectorAlignment) std::array<float, dim> a;
                    std::copy_n(primes.cbegin(), dim, a.begin());
                    return a;
                }(),
                skip_iterations)
        {
        }
        FloatVector<dim> b;
        FloatVector<dim> n = FloatVector<dim>::fill(0);
        FloatVector<dim> d = FloatVector<dim>::fill(1);

        inline auto initialize(std::size_t n) noexcept
        {
            for (auto i = 0U; i < n; ++i)
            {
                next();
            }
        }

        inline auto next() noexcept -> FloatVector<dim>
        {
            auto xf = d - n;
            auto x_eq_1 = xf == 1.;
            auto x_neq_1 = ~x_eq_1;

            // if x == 1
            d = d.blend((d * b).floor(), x_eq_1);

            // if x != 1 (zero out) ignore if x == 1
            auto y = x_neq_1 & (d / b).floor();
            auto x_le_y = x_neq_1 & (xf <= y);

            while (x_le_y.any())
            {
                y = y.blend((y / b).floor(), x_le_y);
                x_le_y = x_le_y & (xf <= y);
            }

            n = (((b + 1.F) * y).floor() - xf).blend(FloatVector<dim>::fill(1), x_eq_1);
            return (n / d).trim();
        }
    };
}  // namespace vamp::rng
