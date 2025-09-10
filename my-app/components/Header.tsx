import React from 'react'
import Image from 'next/image'


const Header = () => {
  return (
    <div className='w-[80%] m-auto my-20 h-[30%] flex flex-col items-center justify-center'>
      {/* Logo Image */}
      <img src="/Images/logo.png" className='w-[150px]'/>
      {/* Heading */}
      <h1 className="text-7xl text-blue-700 font-extrabold mb-4 text-center">SkyPredict</h1>
      {/* Catch Phrase */}
      <p className="text-xl text-gray-600 font-medium text-center">Predict flight prices instantly and travel smarter with AI-powered insights.</p>
    </div>
  )
}

export default Header