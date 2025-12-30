/**
 * Temporary type shim for .jsx files during migration
 */

declare module '*.jsx' {
  import { ComponentType } from 'react';
  const component: ComponentType<any>;
  export default component;
}
