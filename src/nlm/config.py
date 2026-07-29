from dataclasses import dataclass


@dataclass(frozen=True)
class NLMConfig:
    image_size: int = 64
    noise_sigma: float = 0.08
    patch_size: int = 3
    search_window_size: int = 7
    h: float = 0.12
    random_seed: int = 42
    block_size_x: int = 16
    block_size_y: int = 16

    @property
    def patch_radius(self) -> int:
        return self.patch_size // 2

    @property
    def search_radius(self) -> int:
        return self.search_window_size // 2

    @property
    def padding_radius(self) -> int:
        return self.patch_radius + self.search_radius

    @property
    def block_size(self) -> tuple[int, int]:
        return self.block_size_x, self.block_size_y

    def validate(self) -> None:
        if self.image_size <= 0:
            raise ValueError("image_size must be positive.")

        if self.patch_size <= 0 or self.patch_size % 2 == 0:
            raise ValueError(
                "patch_size must be a positive odd integer."
            )

        if (
            self.search_window_size <= 0
            or self.search_window_size % 2 == 0
        ):
            raise ValueError(
                "search_window_size must be a positive odd integer."
            )

        if self.search_window_size < self.patch_size:
            raise ValueError(
                "search_window_size must be at least patch_size."
            )

        if self.noise_sigma < 0:
            raise ValueError("noise_sigma cannot be negative.")

        if self.h <= 0:
            raise ValueError("h must be positive.")

        if self.block_size_x <= 0 or self.block_size_y <= 0:
            raise ValueError("CUDA block dimensions must be positive.")