import * as React from "react";

const Checkbox = React.forwardRef(
  ({ className = "", checked = false, onCheckedChange, ...props }, ref) => {
    const handleChange = (e) => {
      if (onCheckedChange) {
        onCheckedChange(e.target.checked);
      }
    };

    return (
      <input
        type="checkbox"
        ref={ref}
        checked={checked}
        onChange={handleChange}
        className={`h-4 w-4 rounded border-input text-primary focus:ring-2 focus:ring-ring focus:ring-offset-2 disabled:cursor-not-allowed disabled:opacity-50 ${className}`}
        {...props}
      />
    );
  }
);
Checkbox.displayName = "Checkbox";

export { Checkbox };
