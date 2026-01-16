import * as React from "react";

const Slider = React.forwardRef(
  (
    {
      className = "",
      min = 0,
      max = 100,
      step = 1,
      value = [50],
      onValueChange,
      ...props
    },
    ref
  ) => {
    const handleChange = (e) => {
      const newValue = Number(e.target.value);
      if (onValueChange) {
        onValueChange([newValue]);
      }
    };

    return (
      <input
        type="range"
        ref={ref}
        min={min}
        max={max}
        step={step}
        value={value[0]}
        onChange={handleChange}
        className={`w-full h-2 bg-secondary rounded-lg appearance-none cursor-pointer accent-primary ${className}`}
        {...props}
      />
    );
  }
);
Slider.displayName = "Slider";

export { Slider };
