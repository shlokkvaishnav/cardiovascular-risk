"use client";

import { ThemeProvider as NextThemesProvider, type ThemeProviderProps as NextThemesProviderProps } from "next-themes";
import * as React from "react";

export function ThemeProvider({ children, ...props }: NextThemesProviderProps) {
  return <NextThemesProvider {...props}>{children}</NextThemesProvider>;
}
